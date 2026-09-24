//! Native model format compatibility: version-1 files (hessboost 0.1.1) load
//! with their original predictions, version 2 round-trips every model
//! feature, and unknown versions are refused.
//!
//! `tests/data/native-v1/` holds `save_binary` / `save_json` output of
//! hessboost 0.1.1 (commit 1af4e21) plus `predictions.json`, the `f32` bit
//! patterns of `predict` and `predict_margin` on [`features`]. They were
//! produced by running `tests/data/native-v1/generate.rs` as an example of a
//! 0.1.1 checkout:
//!
//! ```sh
//! git worktree add /tmp/hb-v1 1af4e21
//! cp tests/data/native-v1/generate.rs /tmp/hb-v1/examples/gen_native_v1.rs
//! out="$PWD/tests/data/native-v1"
//! (cd /tmp/hb-v1 && cargo run --release --example gen_native_v1 -- "$out")
//! git worktree remove --force /tmp/hb-v1
//! ```

use hessboost::prelude::*;
use serde_json::Value;
use std::path::PathBuf;

const ROWS: usize = 48;
const COLS: usize = 4;
const V1_MODELS: [&str; 7] = [
    "reg",
    "binary",
    "softprob",
    "dart",
    "categorical",
    "gblinear",
    "early_stop",
];

/// The generator's feature matrix: column 0 holds category codes `0..5`,
/// the others values in `[0, 1)`, with every value whose hash is a multiple
/// of 11 missing. Only exact `f32` operations, so every platform agrees.
fn feature(i: usize, j: usize) -> f32 {
    let h = (i as u64 * 2_654_435_761 + j as u64 * 40_503 + 17) % 1009;
    if h.is_multiple_of(11) {
        f32::NAN
    } else if j == 0 {
        (h % 5) as f32
    } else {
        h as f32 / 1009.0
    }
}

fn features() -> Vec<f32> {
    (0..ROWS)
        .flat_map(|i| (0..COLS).map(move |j| feature(i, j)))
        .collect()
}

fn v1_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/native-v1")
}

fn v1_data(name: &str) -> DMatrix {
    let data = DMatrix::from_dense(&features(), ROWS, COLS).unwrap();
    if name == "categorical" {
        let mut types = vec![FeatureType::Numerical; COLS];
        types[0] = FeatureType::Categorical;
        data.with_feature_types(&types).unwrap()
    } else {
        data
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn expected(predictions: &Value, name: &str, kind: &str) -> Vec<u32> {
    predictions[name][kind]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}

/// `model`'s prediction transform applied to `margins` through the current
/// code path: the same model without trees or linear weights, predicting on
/// `data` with `margins` as its per-row-and-output base margin.
fn transform_of(model: &BoostedModel, data: DMatrix, margins: &[f32]) -> Vec<u32> {
    let mut doc: Value = serde_json::from_str(&model.to_json().unwrap()).unwrap();
    doc["trees"] = Value::Array(Vec::new());
    doc["tree_weights"] = Value::Array(Vec::new());
    doc["best_iteration"] = Value::Null;
    doc["linear"] = Value::Null;
    let bare = BoostedModel::from_json(&doc.to_string()).unwrap();
    let data = data.with_base_margin(margins).unwrap();
    bits(&bare.predict(&data).unwrap())
}

/// Margins are sums of stored leaf values, identical on every backend, so
/// they must reproduce 0.1.1 bit for bit. Transformed predictions go through
/// SIMD-dispatched transcendental kernels (e.g. NEON softmax for three
/// classes on aarch64, scalar on x86-64) that differ in the last bits, so
/// `predict` must equal the current transform of those margins exactly and
/// the recorded 0.1.1 predictions within the kernels' tolerance
/// (`simd/tests.rs`), which still refuses a wrong decoded transform.
#[test]
fn v1_files_load_with_their_original_predictions() {
    let dir = v1_dir();
    let predictions: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("predictions.json")).unwrap())
            .unwrap();
    for name in V1_MODELS {
        let bytes = std::fs::read(dir.join(format!("{name}.bin"))).unwrap();
        assert_eq!(&bytes[..5], b"SQB\0\x01", "{name}: not a v1 file");
        let from_binary = BoostedModel::from_bytes(&bytes).unwrap();
        let from_json = BoostedModel::load_json(dir.join(format!("{name}.json"))).unwrap();
        let data = v1_data(name);
        let margin = expected(&predictions, name, "margin");
        let recorded = expected(&predictions, name, "predict");
        for model in [&from_binary, &from_json] {
            assert_eq!(model.n_targets(), 1, "{name}");
            assert_eq!(model.num_parallel_tree(), 1, "{name}");
            assert!(!model.has_vector_leaves(), "{name}");
            assert!(model.trees().iter().all(|t| t.linear_leaves().is_none()));
            assert_eq!(
                bits(&model.predict_margin(&data).unwrap()),
                margin,
                "{name}: predict_margin"
            );
            let predict = bits(&model.predict(&data).unwrap());
            let margin_values: Vec<f32> = margin.iter().map(|&b| f32::from_bits(b)).collect();
            assert_eq!(
                predict,
                transform_of(model, v1_data(name), &margin_values),
                "{name}: predict is not the transform of the margins"
            );
            assert_eq!(predict.len(), recorded.len(), "{name}");
            for (i, (&got, &want)) in predict.iter().zip(&recorded).enumerate() {
                let (got, want) = (f32::from_bits(got), f32::from_bits(want));
                assert!(
                    (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                    "{name}: predict[{i}] {got} differs from the recorded {want}"
                );
            }
        }
        // The binary migration and the JSON serde defaults build the same
        // model, and re-saving it writes the current version.
        let resaved = from_binary.to_bytes().unwrap();
        assert_eq!(resaved[4], 2, "{name}");
        assert_eq!(resaved, from_json.to_bytes().unwrap(), "{name}");
        assert_eq!(
            bits(
                &BoostedModel::from_bytes(&resaved)
                    .unwrap()
                    .predict(&data)
                    .unwrap()
            ),
            bits(&from_binary.predict(&data).unwrap()),
            "{name}: v2 re-save"
        );
    }
}

#[test]
fn v1_layout_metadata_survives_the_migration() {
    let dir = v1_dir();
    let load = |name: &str| BoostedModel::load_binary(dir.join(format!("{name}.bin"))).unwrap();
    // Round-robin multiclass trees become whole iterations of one tree per
    // class; early stopping keeps its round index.
    let softprob = load("softprob");
    assert_eq!(
        (softprob.num_trees(), softprob.trees_per_iteration()),
        (6, 3)
    );
    assert_eq!(softprob.num_boost_rounds(), 2);
    let stopped = load("early_stop");
    assert_eq!(stopped.best_iteration(), Some(0));
    assert_eq!(stopped.num_boost_rounds(), 3);
    assert_eq!(load("binary").objective_params().scale_pos_weight, 2.0);
    let params = load("reg").objective_params().clone();
    assert!(params.quantile_alpha.is_empty() && params.expectile_alpha.is_empty());
    assert_eq!(params.aft_loss_distribution, AftDistribution::Normal);
    assert_eq!(params.aft_loss_distribution_scale, 1.0);
    // Every v1 model, binary or JSON, gets the `dist:*` defaults.
    for name in V1_MODELS {
        let json = BoostedModel::load_json(dir.join(format!("{name}.json"))).unwrap();
        for model in [load(name), json] {
            let p = model.objective_params();
            assert_eq!(p.distribution, None, "{name}");
            assert_eq!(p.dist_gradient, DistGradient::Fisher, "{name}");
            assert_eq!(p.dist_split_direction, DistSplitDirection::Random, "{name}");
        }
    }
}

#[test]
fn unknown_and_corrupt_native_payloads_are_refused() {
    let v1 = std::fs::read(v1_dir().join("reg.bin")).unwrap();
    let v2 = BoostedModel::from_bytes(&v1).unwrap().to_bytes().unwrap();
    for version in [0u8, 3, 255] {
        let mut bytes = v2.clone();
        bytes[4] = version;
        let err = BoostedModel::from_bytes(&bytes).unwrap_err();
        assert!(
            matches!(&err, HessboostError::ModelFormat(msg)
                if msg.contains(&format!("unsupported native model version {version}"))),
            "{err}"
        );
    }
    // Each version decodes only its own layout, and truncation is caught.
    let mut v2_as_v1 = v2.clone();
    v2_as_v1[4] = 1;
    let mut v1_as_v2 = v1.clone();
    v1_as_v2[4] = 2;
    for bytes in [v2_as_v1, v1_as_v2, v1[..v1.len() - 3].to_vec()] {
        assert!(matches!(
            BoostedModel::from_bytes(&bytes),
            Err(HessboostError::ModelFormat(_))
        ));
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

#[test]
fn overflowing_tree_layout_is_refused() {
    // Two outputs × 2^63 parallel trees overflows `usize`: it used to panic
    // (debug) or wrap to zero trees per iteration (release), after which
    // round counts divided by zero.
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
    // A two-alpha objective on a one-output layout used to be accepted and
    // then transform consecutive prediction rows as if they were one row.
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

/// `(x, y)` with `k` label columns: 160 rows, four features, missing values
/// in column 3.
fn train_data(k: usize) -> (Vec<f32>, Vec<f32>) {
    const N: usize = 160;
    let mut x = Vec::with_capacity(N * COLS);
    let mut y = Vec::with_capacity(N * k);
    for i in 0..N {
        let a = ((i * 37) % 101) as f32 / 101.0;
        let b = ((i * 53) % 97) as f32 / 97.0;
        let c = ((i * 11) % 89) as f32 / 89.0;
        let d = if i % 7 == 0 {
            f32::NAN
        } else {
            ((i * 29) % 83) as f32 / 83.0
        };
        x.extend([a, b, c, d]);
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

#[test]
fn v2_round_trips_every_model_feature() {
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
    let softmax = DMatrix::from_dense(&x, n, COLS)
        .unwrap()
        .with_labels(&classes)
        .unwrap();
    let counts: Vec<f32> = (0..n).map(|i| ((i * 7) % 9) as f32).collect();
    let negbinomial = DMatrix::from_dense(&x, n, COLS)
        .unwrap()
        .with_labels(&counts)
        .unwrap();
    let cases: Vec<(&str, TrainingParams, DMatrix, Has)> = vec![
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
    for (name, params, data, has_feature) in cases {
        let model = train(&params, &data, 4).unwrap();
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
