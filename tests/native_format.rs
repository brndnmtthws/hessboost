//! Native model formats: the binary format (a zstd-compressed container of
//! named sections) and native JSON round-trip every model feature bit for
//! bit; models saved by earlier versions in `tests/data/saved/<version>/`
//! keep loading with their recorded margins; and other container versions,
//! corrupt payloads, and inconsistent layouts are refused.
//!
//! Before a release that has no directory there yet, run
//! `cargo test --test native_format -- --ignored save_models_of_this_version`
//! and commit the files it writes. Directories of earlier versions are never
//! regenerated: they are what later versions must keep reading.

use hessboost::prelude::*;
use serde_json::Value;
use std::path::{Path, PathBuf};

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
    // The uncompressed container loads too; its version byte follows the
    // magic.
    let container = zstd::stream::decode_all(bytes.as_slice()).unwrap();
    assert_eq!(&container[..4], b"SQB\0");
    assert_eq!(
        bits(
            &BoostedModel::from_bytes(&container)
                .unwrap()
                .predict(&matrix(1))
                .unwrap()
        ),
        bits(&model.predict(&matrix(1)).unwrap())
    );
    let with_version = |version: u8| {
        let mut bytes = container.clone();
        bytes[4] = version;
        bytes
    };
    // Every container version but the current one is refused, and so are
    // truncated payloads and headers, compressed or not.
    for corrupt in [
        with_version(0),
        with_version(1),
        with_version(2),
        with_version(4),
        with_version(255),
        bytes[..bytes.len() - 3].to_vec(),
        container[..container.len() - 3].to_vec(),
        container[..4].to_vec(),
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

/// One trained model per stored model feature, with the data it is
/// evaluated on. The data (built from [`four_features`]) must never change:
/// the margins saved under `tests/data/saved/` were recorded on it.
fn feature_models() -> Vec<(&'static str, BoostedModel, DMatrix, Has)> {
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
    models
}

#[test]
fn native_formats_round_trip_every_model_feature() {
    for (name, model, data, has_feature) in feature_models() {
        assert!(has_feature(&model), "{name}: feature not exercised");
        let bytes = model.to_bytes().unwrap();
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

/// Where the models saved by hessboost release `version` live.
fn saved_dir(version: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/saved")
        .join(version)
}

/// A file-name form of a [`feature_models`] case name.
fn slug(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn margin_bytes(margins: &[f32]) -> Vec<u8> {
    margins.iter().flat_map(|m| m.to_le_bytes()).collect()
}

/// Every model saved by an earlier (or this) version loads from each format
/// it was saved in and reproduces the margins recorded when it was saved.
#[test]
fn saved_models_keep_loading_with_their_margins() {
    let root = saved_dir("");
    let mut versions: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    versions.sort();
    assert!(
        !versions.is_empty(),
        "no saved models under {}",
        root.display()
    );
    let cases = feature_models();
    for dir in versions {
        for (name, _, data, _) in &cases {
            let file = |ext: &str| dir.join(format!("{}.{ext}", slug(name)));
            let expected = std::fs::read(file("margins")).unwrap();
            let binary = BoostedModel::load_binary(file("bin")).unwrap();
            let json = BoostedModel::load_json(file("json")).unwrap();
            for (format, model) in [("bin", binary), ("json", json)] {
                let margins = margin_bytes(&model.predict_margin(data).unwrap());
                assert!(margins == expected, "{}: {name} ({format})", dir.display());
            }
            if file("hbtd").exists() {
                let compact = CompactModel::load(file("hbtd")).unwrap();
                let margins = margin_bytes(&compact.predict_margin(data).unwrap());
                assert!(margins == expected, "{}: {name} (compact)", dir.display());
            }
        }
    }
}

/// Write this version's saved models (see the module docs). Refuses to
/// overwrite a version's existing directory.
#[test]
#[ignore = "run once per release, then commit tests/data/saved/<version>"]
fn save_models_of_this_version() {
    let dir = saved_dir(env!("CARGO_PKG_VERSION"));
    assert!(
        !dir.exists(),
        "{} exists; saved models of a version are never rewritten",
        dir.display()
    );
    std::fs::create_dir_all(&dir).unwrap();
    for (name, model, data, _) in feature_models() {
        let file = |ext: &str| dir.join(format!("{}.{ext}", slug(name)));
        model.save_binary(file("bin")).unwrap();
        model.save_json(file("json")).unwrap();
        if let Ok(compact) = model.to_compact() {
            compact.save(file("hbtd")).unwrap();
        }
        let margins = model.predict_margin(&data).unwrap();
        std::fs::write(file("margins"), margin_bytes(&margins)).unwrap();
    }
}
