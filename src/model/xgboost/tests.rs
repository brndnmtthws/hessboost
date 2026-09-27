//! Tests of the XGBoost JSON/UBJSON interchange.

use super::*;
use super::{objective::*, parse::*, tree::*};
use crate::config::{BoosterKind, Dart, MaxDeltaStep, TrainingParams};
use crate::data::{DMatrix, FeatureType};
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::model::ModelSpec;
use crate::model::objective::ModelObjective;
use crate::objective::AftDistribution;
use crate::objective::{Aft, LambdaRank, PseudoHuber, RegLoss, Tweedie};
use crate::objective::{Objective, Quantiles};
use crate::test_support::labeled_dense;
use crate::training::train;
use serde_json::{Value, json};

/// Train a small squared-error model on a noisy nonlinear signal.
fn reg_model() -> (BoostedModel, DMatrix) {
    let n = 120;
    let mut x = Vec::with_capacity(n * 2);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = i as f32 / n as f32;
        let b = ((i * 7) % n) as f32 / n as f32;
        x.push(a);
        x.push(b);
        y.push(2.0 * a - 3.0 * b + if a > 0.5 { 1.0 } else { -1.0 });
    }
    let d = labeled_dense(&x, n, 2, &y);
    let params = TrainingParams::builder()
        .objective(Objective::SquaredError(RegLoss::default()))
        .max_depth(3)
        .eta(0.3)
        .build()
        .unwrap();
    (train(&params, &d, 15).unwrap(), d)
}

/// A DART model on `d` with non-unit tree weights.
fn dart_model(d: &DMatrix) -> BoostedModel {
    let params = TrainingParams::builder()
        .booster(BoosterKind::Dart(
            Dart::builder().rate_drop(0.5).build().unwrap(),
        ))
        .max_depth(3)
        .build()
        .unwrap();
    train(&params, d, 8).unwrap()
}

/// A shallow model on one categorical feature, and its data.
fn categorical_model() -> (BoostedModel, DMatrix) {
    let categorical = labeled_dense(
        &[0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        6,
        1,
        &[1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    )
    .with_feature_types(&[FeatureType::Categorical])
    .unwrap();
    let params = TrainingParams::builder().max_depth(2).build().unwrap();
    (train(&params, &categorical, 3).unwrap(), categorical)
}

/// The XGBoost JSON export of `model`, as text and parsed.
fn export_json_document(model: &BoostedModel) -> (String, Value) {
    let text = export_xgboost_json(model).unwrap();
    let json = serde_json::from_str(&text).unwrap();
    (text, json)
}

/// Assert `result` is a [`HessboostError::ModelFormat`]; `context` labels
/// a failure.
fn assert_format_error<T: std::fmt::Debug>(result: Result<T>, context: impl std::fmt::Display) {
    let err = result.unwrap_err();
    assert!(
        matches!(err, HessboostError::ModelFormat(_)),
        "{context}: {err}"
    );
}

#[test]
fn roundtrip_reg_preserves_predictions() {
    let (model, d) = reg_model();
    let before = model.predict(&d).unwrap();

    let json = export_xgboost_json(&model).unwrap();
    let restored = import_xgboost_json(&json).unwrap();
    let after = restored.predict(&d).unwrap();

    assert_eq!(restored.num_trees(), model.num_trees());
    assert_eq!(restored.n_features(), model.n_features());
    assert_eq!(restored.objective(), model.objective());
    assert_eq!(
        (before.n_rows(), before.width()),
        (after.n_rows(), after.width())
    );
    for (a, b) in before.as_slice().iter().zip(after.as_slice()) {
        assert!((a - b).abs() < 1e-5, "pred drift: {a} vs {b}");
    }
}

#[test]
fn feature_counts_round_trip_exactly() {
    // `num_feature` is written as an integer string; reading it through
    // `f64` rounded counts above 2^53 to a neighbor.
    let (model, _) = reg_model();
    for n_features in [(1usize << 53) + 1, usize::MAX] {
        let mut wide = model.clone();
        wide.n_features = n_features;
        for restored in [
            import_xgboost_json(&export_xgboost_json(&wide).unwrap()).unwrap(),
            import_xgboost_ubjson(&export_xgboost_ubjson(&wide).unwrap()).unwrap(),
        ] {
            assert_eq!(restored.n_features(), n_features);
        }
    }
    let lmp = |v: &str| serde_json::json!({ "num_feature": v });
    assert_eq!(count_param(&lmp("4.0"), "num_feature", 0).unwrap(), 4);
    assert!(count_param(&lmp("4.5"), "num_feature", 0).is_err());
    assert!(count_param(&lmp("-1"), "num_feature", 0).is_err());
}

#[test]
fn roundtrip_binary_preserves_predictions() {
    // Binary logistic exercises the prob<->margin base_score link.
    let n = 80;
    let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
    let y: Vec<f32> = x.iter().map(|&v| f32::from(v > 0.4)).collect();
    let d = labeled_dense(&x, n, 1, &y);
    let params = TrainingParams::builder()
        .objective(Objective::BinaryLogistic(RegLoss::default()))
        .max_depth(3)
        .eta(0.3)
        .build()
        .unwrap();
    let model = train(&params, &d, 20).unwrap();
    let before = model.predict(&d).unwrap();

    let json = export_xgboost_json(&model).unwrap();
    let restored = import_xgboost_json(&json).unwrap();
    assert_eq!(restored.objective().name(), "binary:logistic");
    // base_score should round-trip through the logit/sigmoid link.
    assert!((restored.base_score() - model.base_score()).abs() < 1e-4);
    let after = restored.predict(&d).unwrap();
    for (a, b) in before.as_slice().iter().zip(after.as_slice()) {
        assert!((a - b).abs() < 1e-5, "pred drift: {a} vs {b}");
    }
}

/// A minimal, hand-written XGBoost 3.x stump: feature 0 with threshold
/// 1.5, left leaf +10, right leaf -10, `base_score` `[0]` (raw margin).
/// `objective`'s document is the 3.4.1 shape for `reg:squarederror`.
fn hand_stump() -> Value {
    json!({
        "version": [3, 4, 1],
        "learner": {
            "gradient_booster": {
                "name": "gbtree",
                "model": {
                    "gbtree_model_param": {"num_parallel_tree": "1", "num_trees": "1"},
                    "iteration_indptr": [0, 1],
                    "tree_info": [0],
                    "trees": [{
                        "id": 0,
                        "tree_param": {"num_nodes": "3", "num_feature": "1", "size_leaf_vector": "1"},
                        "left_children": [1, -1, -1],
                        "right_children": [2, -1, -1],
                        "parents": [2_147_483_647, 0, 0],
                        "split_indices": [0, 0, 0],
                        "split_conditions": [1.5, 10.0, -10.0],
                        "default_left": [1, 0, 0],
                        "base_weights": [0.0, 10.0, -10.0],
                        "loss_changes": [42.0, 0.0, 0.0],
                        "sum_hessian": [8.0, 5.0, 3.0],
                        "split_type": [0, 0, 0]
                    }]
                }
            },
            "learner_model_param": {
                "base_score": "[0E0]", "boost_from_average": "1",
                "num_class": "0", "num_feature": "1", "num_target": "1"
            },
            "objective": {"name": "reg:squarederror", "reg_loss_param": {"scale_pos_weight": "1"}}
        }
    })
}

/// Import `doc` through its JSON text.
fn import_doc(doc: &Value) -> Result<BoostedModel> {
    import_xgboost_json(&doc.to_string())
}

/// `doc`'s `learner_model_param` block.
fn model_param(doc: &mut Value) -> &mut Value {
    &mut doc["learner"]["learner_model_param"]
}

/// `doc`'s `gradient_booster.model` block.
fn booster_model(doc: &mut Value) -> &mut Value {
    &mut doc["learner"]["gradient_booster"]["model"]
}

#[test]
fn import_hand_written_stump_routes_correctly() {
    let model = import_doc(&hand_stump()).unwrap();
    assert_eq!(model.num_trees(), 1);
    assert_eq!(model.n_features(), 1);
    assert_eq!(model.base_score(), 0.0);

    // x=1.0 (< 1.5) -> left leaf +10 ; x=2.0 (>= 1.5) -> right leaf -10.
    let d = DMatrix::from_dense(&[1.0, 2.0], 2, 1).unwrap();
    let margins = model.predict_margin(&d).unwrap().into_vec(); // one per row
    assert!((margins[0] - 10.0).abs() < 1e-6, "got {}", margins[0]);
    assert!((margins[1] + 10.0).abs() < 1e-6, "got {}", margins[1]);

    // Missing value follows default_left = true -> left leaf.
    let dm = DMatrix::from_dense(&[f32::NAN], 1, 1).unwrap();
    let mm = model.predict_margin(&dm).unwrap().into_vec();
    assert!(
        (mm[0] - 10.0).abs() < 1e-6,
        "missing routed wrong: {}",
        mm[0]
    );
}

/// Three constant stumps, one per class (`tree_info` round-robin), whose
/// leaves are all zero, so `predict_margin` exposes the imported
/// per-class intercepts. `multi:softprob` stores margins directly, so the
/// vector must come through unchanged.
fn three_class(base_score: &str) -> Value {
    let mut doc = parallel_tree(1, &[0, 1, 2], &[0.0; 3], None, None);
    model_param(&mut doc)["base_score"] = json!(base_score);
    doc
}

#[test]
fn import_multiclass_vector_intercept_offsets_each_class() {
    let model = import_doc(&three_class("[5.3293586E-2,-1.3475811E-1,8.146441E-2]")).unwrap();
    let expected = [5.329_358_6E-2f32, -1.347_581_1E-1, 8.146_441E-2];
    assert_eq!(model.base_scores(), &expected);
    let d = DMatrix::from_dense(&[0.0, 1.0], 2, 1).unwrap();
    let margins = model.predict_margin(&d).unwrap();
    assert_eq!(margins.as_slice(), [expected, expected].concat());

    // A single entry applies to every class (XGBoost's old-format rule).
    let uniform = import_doc(&three_class("[5E-1]")).unwrap();
    assert_eq!(uniform.base_scores(), &[0.5, 0.5, 0.5]);
}

#[test]
fn malformed_base_score_is_rejected() {
    for bad in ["0.5", "[0.1,0.2]", "[a]", "[]", "[0.1,0.2,0.3,0.4]"] {
        assert_format_error(import_doc(&three_class(bad)), bad);
    }
}

/// A two-target model with one single-leaf vector tree of width
/// `width` and the given `leaf_weights`.
fn vector_stump(width: &str, leaf_weights: &[f32]) -> Value {
    let mut doc = hand_stump();
    model_param(&mut doc)["num_target"] = json!("2");
    booster_model(&mut doc)["trees"] = json!([{
        "id": 0,
        "tree_param": {"num_nodes": "1", "num_feature": "1", "size_leaf_vector": width},
        "left_children": [-1], "right_children": [0], "parents": [-1],
        "split_indices": [0], "split_conditions": [0.0], "default_left": [0],
        "base_weights": [], "leaf_weights": leaf_weights,
        "loss_changes": [0.0], "sum_hessian": [1.0], "split_type": [0]
    }]);
    booster_model(&mut doc)
        .as_object_mut()
        .unwrap()
        .remove("iteration_indptr");
    doc
}

#[test]
fn vector_leaf_width_is_validated_before_allocating() {
    let model = import_doc(&vector_stump("2", &[1.0, 2.0])).unwrap();
    let d = DMatrix::from_dense(&[0.0], 1, 1).unwrap();
    assert_eq!(model.predict_margin(&d).unwrap().as_slice(), [1.0, 2.0]);
    // A width that saturates `usize` must not panic allocating the leaf
    // storage: it, a width other than the model's outputs, a fractional
    // width, or one the leaf weights cannot fill is a format error.
    for (width, weights) in [
        ("1e30", &[][..]),
        ("18446744073709551615", &[]),
        ("3", &[1.0, 2.0, 3.0]),
        ("2.5", &[1.0, 2.0, 3.0]),
        ("2", &[1.0]),
    ] {
        assert_format_error(import_doc(&vector_stump(width, weights)), width);
    }
}

#[test]
fn vector_leaf_storage_is_bounded_by_the_leaf_weights() {
    // 65,536 outputs over 65,535 nodes would expand to ~16 GiB of leaf
    // storage, so the node graph and leaf mapping must be checked before
    // any of it is allocated.
    const K: usize = 1 << 16;
    const N: usize = K - 1;
    let one_vector = vec![0.5f32; K];
    let tree = |left: Vec<i64>, right: Vec<i64>, leaf_weights: &[f32]| {
        json!({
            "tree_param": {"num_nodes": N.to_string(), "num_feature": "1",
                           "size_leaf_vector": K.to_string()},
            "left_children": left,
            "right_children": right,
            "split_conditions": vec![0.0f32; N],
            "leaf_weights": leaf_weights,
        })
    };
    // Heap-shaped binary tree: internal node `i` has children
    // `2i + 1, 2i + 2`; leaves are numbered in node order.
    let internal = N / 2;
    let heap_left: Vec<i64> = (0..N)
        .map(|i| if i < internal { 2 * i as i64 + 1 } else { -1 })
        .collect();
    let heap_right: Vec<i64> = (0..N)
        .map(|i| {
            if i < internal {
                2 * i as i64 + 2
            } else {
                (i - internal) as i64
            }
        })
        .collect();
    let cases = [
        // Every node a leaf sharing the one serialized vector.
        tree(vec![-1; N], vec![0; N], &one_vector),
        // Binary-tree shape, but internal children out of range.
        tree(
            heap_left
                .iter()
                .map(|&l| if l < 0 { l } else { i64::from(i32::MAX) })
                .collect(),
            heap_right.clone(),
            &one_vector,
        ),
        // Valid graph, but one serialized vector for 32,768 leaves.
        tree(heap_left.clone(), heap_right.clone(), &one_vector),
    ];
    for (case, tj) in cases.iter().enumerate() {
        assert_format_error(tree_from_json(tj, K), case);
    }
    // The same valid graph with a vector per leaf decodes, each leaf
    // reading its own vector.
    let k = 2;
    let leaves = N - internal;
    let weights: Vec<f32> = (0..leaves * k).map(|v| v as f32).collect();
    let mut tj = tree(heap_left, heap_right, &weights);
    tj["tree_param"]["size_leaf_vector"] = json!(k.to_string());
    let decoded = tree_from_json(&tj, k).unwrap();
    assert_eq!(decoded.leaf_vector(internal), [0.0, 1.0]);
    assert_eq!(
        decoded.leaf_vector(N - 1),
        [(2 * leaves - 2) as f32, (2 * leaves - 1) as f32]
    );
}

/// [`hand_stump`] without its tree.
fn treeless() -> Value {
    let mut doc = hand_stump();
    let model = booster_model(&mut doc);
    model["gbtree_model_param"]["num_trees"] = json!("0");
    model["tree_info"] = json!([]);
    model["trees"] = json!([]);
    model.as_object_mut().unwrap().remove("iteration_indptr");
    doc
}

#[test]
fn output_count_is_validated_before_allocating() {
    // `num_target` saturating `usize` must not panic sizing the per-output
    // tree groups or broadcasting the intercept.
    let stump = hand_stump();
    let treeless = treeless();
    assert_eq!(import_doc(&treeless).unwrap().num_trees(), 0);
    let with_targets = |doc: &Value, count: &str| {
        let mut doc = doc.clone();
        model_param(&mut doc)["num_target"] = json!(count);
        doc
    };
    for doc in [&stump, &treeless] {
        for count in ["1e30", "18446744073709551615", "1e18", "2.5", "-1", "0"] {
            assert_format_error(import_doc(&with_targets(doc, count)), count);
        }
    }
    // One tree cannot cover two outputs' groups.
    assert_format_error(import_doc(&with_targets(&stump, "2")), "two outputs");
}

#[test]
fn treeless_intercept_broadcast_is_bounded() {
    // A tree-less document declares its outputs without backing them:
    // broadcasting one `base_score` entry to 2^29 of them would allocate
    // 2 GiB.
    let declare = |key: &str, count: usize| {
        let mut doc = treeless();
        model_param(&mut doc)[key] = json!(count.to_string());
        doc
    };
    let huge = declare("num_target", 536_870_912);
    assert_format_error(import_doc(&huge), "num_target 2^29");
    let mut classes = declare("num_class", 536_870_912);
    classes["learner"]["objective"] = json!({
        "name": "multi:softprob",
        "softmax_multiclass_param": {"num_class": "536870912"}
    });
    assert_format_error(import_doc(&classes), "num_class 2^29");
    // Within the bound, and for an explicit per-output vector, the entry
    // still applies to every output.
    let wide = import_doc(&declare("num_target", MAX_BROADCAST_OUTPUTS)).unwrap();
    assert_eq!(wide.base_scores(), vec![0.0; MAX_BROADCAST_OUTPUTS]);
    let mut listed = declare("num_target", MAX_BROADCAST_OUTPUTS + 1);
    model_param(&mut listed)["base_score"] =
        json!(format_float_vector(vec![0.25; MAX_BROADCAST_OUTPUTS + 1]));
    let listed = import_doc(&listed).unwrap();
    assert_eq!(listed.base_scores(), vec![0.25; MAX_BROADCAST_OUTPUTS + 1]);
}

/// A one-split categorical stump whose categorical nodes take the
/// segments `(begin, size)` of `categories`: node 0 splits to node 1 and
/// leaf 2, node 1 to leaves 3 and 4.
fn categorical_segments(categories: &[u32], segments: [(usize, usize); 2]) -> Value {
    let mut doc = hand_stump();
    booster_model(&mut doc)["trees"] = json!([{
        "id": 0,
        "tree_param": {"num_nodes": "5", "num_feature": "1", "size_leaf_vector": "1"},
        "left_children": [1, 3, -1, -1, -1],
        "right_children": [2, 4, -1, -1, -1],
        "split_indices": [0, 0, 0, 0, 0],
        "split_conditions": [0.0, 0.0, 1.0, 2.0, 3.0],
        "default_left": [0, 0, 0, 0, 0],
        "split_type": [1, 1, 0, 0, 0],
        "categories": categories,
        "categories_nodes": [0, 1],
        "categories_segments": [segments[0].0, segments[1].0],
        "categories_sizes": [segments[0].1, segments[1].1]
    }]);
    doc
}

#[test]
fn categorical_segments_cannot_expand_the_category_array() {
    // Each node copies its segment, so segments overlapping each other
    // would let `n` nodes expand the array `n`-fold.
    let disjoint = categorical_segments(&[0, 1, 1, 2], [(0, 2), (2, 2)]);
    let model = import_doc(&disjoint).unwrap();
    assert_eq!(model.trees()[0].categories().len(), 4);
    let overlapping = categorical_segments(&[0, 1], [(0, 2), (0, 2)]);
    assert_format_error(import_doc(&overlapping), "overlapping segments");
}

#[test]
fn present_but_invalid_objective_parameters_are_refused() {
    let with_objective = |objective: &str| {
        let mut doc = hand_stump();
        doc["learner"]["objective"] = serde_json::from_str(objective).unwrap();
        model_param(&mut doc)["base_score"] = json!("[5E-1]");
        doc
    };
    // Values that do not parse are refused in any block, read or not;
    // the objective's own parameters must also be valid.
    for objective in [
        r#"{"name": "survival:aft", "aft_loss_param": {"aft_loss_distribution": "unsupported"}}"#,
        r#"{"name": "survival:aft", "aft_loss_param": {"aft_loss_distribution": 1}}"#,
        r#"{"name": "survival:aft", "aft_loss_param": {"aft_loss_distribution_scale": "wide"}}"#,
        r#"{"name": "survival:aft", "aft_loss_param": {"aft_loss_distribution_scale": "0"}}"#,
        r#"{"name": "survival:aft", "aft_loss_param": "normal"}"#,
        r#"{"name": "reg:squarederror", "reg_loss_param": {"scale_pos_weight": "heavy"}}"#,
        r#"{"name": "count:poisson", "poisson_regression_param": {"max_delta_step": null}}"#,
        r#"{"name": "rank:ndcg", "lambdarank_param": {"lambdarank_num_pair_per_sample": "2.5"}}"#,
        r#"{"name": "rank:ndcg", "lambdarank_param": {"lambdarank_num_pair_per_sample": "0"}}"#,
        r#"{"name": "reg:quantileerror", "quantile_loss_param": {"quantile_alpha": 0.5}}"#,
        r#"{"name": "reg:quantileerror", "quantile_loss_param": {"quantile_alpha": "[a]"}}"#,
        r#"{"name": "binary:logistic", "reg_loss_param": {"scale_pos_weight": "-1"}}"#,
        r#"{"name": "reg:gamma", "reg_loss_param": {"scale_pos_weight": "-1"}}"#,
        r#"{"name": 7}"#,
        r#""reg:squarederror""#,
    ] {
        assert_format_error(import_doc(&with_objective(objective)), objective);
    }
    // Every `RegLossObj` objective keeps its `scale_pos_weight`;
    // parameters the objective does not read are dropped, whatever
    // their (parseable) value.
    let reweighted = RegLoss::new(3.0).unwrap();
    for (objective, expected) in [
        (
            r#"{"name": "reg:squarederror", "reg_loss_param": {"scale_pos_weight": "3"}}"#,
            Objective::SquaredError(reweighted),
        ),
        (
            r#"{"name": "reg:gamma", "reg_loss_param": {"scale_pos_weight": "3"}}"#,
            Objective::Gamma(reweighted),
        ),
        (
            r#"{"name": "reg:squarederror", "tweedie_regression_param": {"tweedie_variance_power": "5"}}"#,
            Objective::SquaredError(RegLoss::default()),
        ),
        (
            r#"{"name": "reg:squaredlogerror", "reg_loss_param": {"scale_pos_weight": "3"}}"#,
            Objective::SquaredLogError,
        ),
    ] {
        let model = import_doc(&with_objective(objective)).unwrap();
        assert_eq!(model.objective().built_in(), Some(&expected), "{objective}");
    }
    // Genuinely missing blocks and fields keep XGBoost's defaults.
    for objective in [
        r#"{"name": "survival:aft"}"#,
        r#"{"name": "survival:aft", "aft_loss_param": {}}"#,
    ] {
        let model = import_doc(&with_objective(objective)).unwrap();
        assert_eq!(
            model.objective().built_in(),
            Some(&Objective::Aft(Aft::default()))
        );
    }
    let unset = with_objective(
        r#"{"name": "rank:ndcg", "lambdarank_param": {"lambdarank_num_pair_per_sample": "4294967295"}}"#,
    );
    let model = import_doc(&unset).unwrap();
    assert_eq!(
        model.objective().built_in(),
        Some(&Objective::RankNdcg(LambdaRank::default()))
    );
}

#[test]
fn export_refuses_num_class_beside_several_outputs() {
    // XGBoost has no model with `num_class` 2 and two binary targets
    // (`LearnerModelParam` refuses `num_class > 1` with `num_target >
    // 1`); exporting one wrote its margins as probabilities.
    let model = BoostedModel::from_parts(
        Vec::new(),
        Vec::new(),
        vec![0.0, 0.0],
        ModelSpec {
            objective: ModelObjective::trained_with(&Objective::BinaryLogistic(RegLoss::default())),
            max_delta_step: 0.0,
            num_class: 2,
            n_outputs: 2,
            n_targets: 2,
            n_features: 1,
        },
    );
    assert_format_error(export_xgboost_json(&model), "num_class with two targets");
    assert_format_error(export_xgboost_ubjson(&model), "num_class with two targets");
}

#[test]
fn export_writes_xgboost_3_learner_params() {
    let (model, _) = reg_model();
    let (_, json) = export_json_document(&model);
    assert_eq!(json["version"], json!([3, 4, 2]));
    let lmp = &json["learner"]["learner_model_param"];
    assert_eq!(lmp["boost_from_average"], "0");
    assert_eq!(lmp["num_target"], "1");
    let expected = format!("[{}]", model.base_score());
    assert_eq!(lmp["base_score"], expected);
    assert_eq!(
        json["learner"]["objective"],
        json!({"name": "reg:squarederror", "reg_loss_param": {"scale_pos_weight": "1"}})
    );
    assert!(
        json["learner"]["gradient_booster"]["model"]
            .get("weight_drop")
            .is_none()
    );
}

#[test]
fn unsupported_booster_is_rejected() {
    let js = r#"{"learner": {"gradient_booster": {"name": "gblinear"},
                 "learner_model_param": {"num_feature": "3", "base_score": "[0]"}}}"#;
    assert_format_error(import_xgboost_json(js), "gblinear");
}

#[test]
fn dart_roundtrips_through_weight_drop() {
    let (_, d) = reg_model();
    let model = dart_model(&d);
    assert!(model.has_non_unit_tree_weights());
    let before = model.predict(&d).unwrap();

    let (exported, json) = export_json_document(&model);
    assert_eq!(json["learner"]["gradient_booster"]["name"], "gbtree");
    let weight_drop = json["learner"]["gradient_booster"]["model"]["weight_drop"]
        .as_array()
        .unwrap();
    assert_eq!(weight_drop.len(), model.num_trees());

    let restored = import_xgboost_json(&exported).unwrap();
    for t in 0..model.num_trees() {
        assert_eq!(restored.tree_weight(t), model.tree_weight(t), "tree {t}");
    }
    assert_eq!(restored.predict(&d).unwrap(), before);
}

#[test]
fn gblinear_export_is_rejected_and_categorical_roundtrips() {
    let (_, d) = reg_model();
    let params = TrainingParams::builder()
        .booster(BoosterKind::GbLinear)
        .build()
        .unwrap();
    let model = train(&params, &d, 3).unwrap();
    assert!(export_xgboost_json(&model).is_err());

    let (model, categorical) = categorical_model();
    assert!(model.trees().iter().any(|tree| tree.node(0).is_categorical));
    let before = model.predict(&categorical).unwrap();
    let restored = import_xgboost_json(&export_xgboost_json(&model).unwrap()).unwrap();
    assert_eq!(restored.predict(&categorical).unwrap(), before);
}

/// XGBoost 3.4.2 `save_raw("ubj")` / `save_raw("json")` of one booster
/// with categorical splits and missing values, generated by:
///
/// ```python
/// rng = np.random.default_rng(7)
/// x = rng.random((256, 3), dtype=np.float32)
/// x[:, 0] = rng.integers(0, 6, 256)
/// x[rng.random(x.shape) < 0.1] = np.nan
/// y = ((x[:, 0] % 2 == 1) ^ (x[:, 1] > 0.5)).astype(np.float32)
/// d = xgb.DMatrix(x, label=y, feature_types=["c", "q", "q"], enable_categorical=True)
/// b = xgb.train({"max_depth": 2, "objective": "binary:logistic", "nthread": 1,
///                "max_cat_to_onehot": 1}, d, num_boost_round=3)
/// ```
const XGB_UBJ: &[u8] = include_bytes!("../../../tests/data/xgboost-3.4.2-categorical.ubj");
const XGB_JSON: &str = include_str!("../../../tests/data/xgboost-3.4.2-categorical.json");

#[test]
fn xgboost_ubjson_reencodes_byte_for_byte() {
    // Decoding XGBoost's own bytes and encoding them again with the typed
    // array table reproduces the file exactly: same markers, integer
    // widths, typed element types, key order and big-endian payloads.
    let document = ubjson::decode(XGB_UBJ).unwrap();
    let reencoded = ubjson::encode(&document, &xgboost_typed_array).unwrap();
    assert!(reencoded == XGB_UBJ, "re-encoded XGBoost UBJSON differs");
}

#[test]
fn xgboost_ubjson_imports_like_its_json_twin() {
    let from_ubj = import_xgboost_ubjson(XGB_UBJ).unwrap();
    let from_json = import_xgboost_json(XGB_JSON).unwrap();
    assert!(
        from_ubj
            .trees()
            .iter()
            .any(|t| t.nodes().iter().any(|n| n.is_categorical && !n.is_leaf()))
    );
    assert_eq!(from_ubj.to_bytes().unwrap(), from_json.to_bytes().unwrap());

    let truncated = &XGB_UBJ[..XGB_UBJ.len() / 2];
    assert_format_error(import_xgboost_ubjson(truncated), "truncated");
}

#[test]
fn ubjson_export_is_the_json_document_with_typed_tree_arrays() {
    let (reg, d) = reg_model();
    let dart = dart_model(&d);
    let (categorical_model, categorical) = categorical_model();
    for (model, data) in [(reg, &d), (dart, &d), (categorical_model, &categorical)] {
        let ubj = export_xgboost_ubjson(&model).unwrap();
        // The very document the JSON export prints (compared before text
        // formatting, which `serde_json` does not round-trip bit-exactly).
        assert_eq!(
            ubjson::decode(&ubj).unwrap(),
            model_to_value(&model).unwrap()
        );
        for header in [
            &b"split_conditions[$d#L"[..],
            b"base_weights[$d#L",
            b"loss_changes[$d#L",
            b"sum_hessian[$d#L",
            b"left_children[$l#L",
            b"right_children[$l#L",
            b"parents[$l#L",
            b"split_indices[$l#L",
            b"categories[$l#L",
            b"categories_nodes[$l#L",
            b"default_left[$U#L",
            b"split_type[$U#L",
            b"categories_segments[$L#L",
            b"categories_sizes[$L#L",
        ] {
            assert!(
                ubj.windows(header.len()).any(|w| w == header),
                "missing {}",
                String::from_utf8_lossy(header)
            );
        }
        let restored = import_xgboost_ubjson(&ubj).unwrap();
        assert_eq!(
            restored.predict(data).unwrap(),
            model.predict(data).unwrap()
        );
        let via_json = import_xgboost_json(&export_xgboost_json(&model).unwrap()).unwrap();
        assert_eq!(restored.to_bytes().unwrap(), via_json.to_bytes().unwrap());
    }
}

/// A 3-class `gbtree` document of constant stumps whose leaf values are
/// `leaves`, tagged with `tree_info`, laid out with `num_parallel_tree`
/// parallel trees and optional `iteration_indptr` / `weight_drop` arrays.
fn parallel_tree(
    num_parallel_tree: usize,
    tree_info: &[usize],
    leaves: &[f32],
    iteration_indptr: Option<&[usize]>,
    weight_drop: Option<&[f32]>,
) -> Value {
    let trees: Vec<Value> = leaves
        .iter()
        .enumerate()
        .map(|(id, leaf)| {
            json!({
                "id": id,
                "tree_param": {"num_nodes": "1", "num_feature": "1", "size_leaf_vector": "1"},
                "left_children": [-1], "right_children": [-1], "parents": [2_147_483_647],
                "split_indices": [0], "split_conditions": [leaf], "default_left": [0],
                "base_weights": [leaf], "loss_changes": [0.0], "sum_hessian": [1.0],
                "split_type": [0]
            })
        })
        .collect();
    let mut doc = hand_stump();
    let model = booster_model(&mut doc);
    *model = json!({
        "gbtree_model_param": {
            "num_parallel_tree": num_parallel_tree.to_string(),
            "num_trees": leaves.len().to_string()
        },
        "tree_info": tree_info,
        "trees": trees,
    });
    if let Some(indptr) = iteration_indptr {
        model["iteration_indptr"] = json!(indptr);
    }
    if let Some(weights) = weight_drop {
        model["weight_drop"] = json!(weights);
    }
    let param = model_param(&mut doc);
    param["num_class"] = json!("3");
    doc["learner"]["objective"] =
        json!({"name": "multi:softprob", "softmax_multiclass_param": {"num_class": "3"}});
    doc
}

/// Per-class margins of a one-row prediction through `doc`.
fn class_margins(doc: &Value) -> Vec<f32> {
    let model = import_doc(doc).unwrap();
    let d = DMatrix::from_dense(&[0.0], 1, 1).unwrap();
    model.predict_margin(&d).unwrap().into_vec()
}

#[test]
fn import_keeps_parallel_trees_as_iterations() {
    // One iteration, two parallel trees per class: XGBoost order is
    // [c0, c0, c1, c1, c2, c2]; each class must receive its own leaves.
    let leaves = [1.0, 2.0, 10.0, 20.0, 100.0, 200.0];
    let tree_info = [0, 0, 1, 1, 2, 2];
    let derived = parallel_tree(2, &tree_info, &leaves, None, None);
    assert_eq!(class_margins(&derived), [3.0, 30.0, 300.0]);
    let model = import_doc(&derived).unwrap();
    assert_eq!(
        (model.num_parallel_tree(), model.num_boost_rounds()),
        (2, 1)
    );

    // Two iterations marked explicitly by `iteration_indptr`; the first
    // one's trees are tagged out of group order and get regrouped.
    let leaves2 = [
        [2.0, 1.0, 10.0, 200.0, 20.0, 100.0],
        [4.0, 8.0, 40.0, 80.0, 400.0, 800.0],
    ]
    .concat();
    let tree_info2 = [[0, 0, 1, 2, 1, 2], tree_info].concat();
    let explicit = parallel_tree(2, &tree_info2, &leaves2, Some(&[0, 6, 12]), None);
    assert_eq!(class_margins(&explicit), [15.0, 150.0, 1500.0]);
    let model = import_doc(&explicit).unwrap();
    assert_eq!(model.num_boost_rounds(), 2);
    let first = model.slice(0..1, 1).unwrap();
    let d = DMatrix::from_dense(&[0.0], 1, 1).unwrap();
    assert_eq!(
        first.predict_margin(&d).unwrap().as_slice(),
        [3.0, 30.0, 300.0]
    );

    // DART weights are indexed by XGBoost position and follow their trees.
    let weights = [1.0, 0.5, 1.0, 0.5, 1.0, 0.5];
    let dart = parallel_tree(2, &tree_info, &leaves, None, Some(&weights));
    assert_eq!(class_margins(&dart), [2.0, 20.0, 200.0]);
    // Re-export writes the forest back unchanged.
    let model = import_doc(&dart).unwrap();
    let (_, doc) = export_json_document(&model);
    let booster = &doc["learner"]["gradient_booster"]["model"];
    assert_eq!(booster["gbtree_model_param"]["num_parallel_tree"], "2");
    assert_eq!(booster["tree_info"], json!(tree_info));
    assert_eq!(booster["iteration_indptr"], json!([0, 6]));
    assert_eq!(booster["weight_drop"], json!(weights));

    // Iterations of different forest sizes are unmappable.
    let uneven_info = [0, 0, 1, 2, 1, 2, 0, 1, 2];
    let uneven = parallel_tree(2, &uneven_info, &leaves2[..9], Some(&[0, 6, 9]), None);
    assert_format_error(import_doc(&uneven), "uneven");

    // Groups with unequal tree counts in one iteration are unmappable.
    let lopsided = parallel_tree(2, &[0, 0, 1, 2, 2, 0], &leaves, None, None);
    assert_format_error(import_doc(&lopsided), "lopsided");
    let mut missing = parallel_tree(1, &tree_info, &leaves, None, None);
    booster_model(&mut missing)
        .as_object_mut()
        .unwrap()
        .remove("tree_info");
    assert_format_error(import_doc(&missing), "no tree_info");
}

#[test]
fn objective_params_roundtrip_through_parameter_blocks() {
    /// Reads the retained value of one case's parameter.
    type Retained = fn(&BoostedModel) -> Option<f64>;
    let n = 40;
    let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
    let counts: Vec<f32> = (0..n).map(|i| (i % 4) as f32).collect();
    let binary: Vec<f32> = x.iter().map(|&v| f32::from(v > 0.6)).collect();
    let d = labeled_dense(&x, n, 1, &counts);
    let positive: Vec<f32> = counts.iter().map(|c| c + 1.0).collect();
    let positive = labeled_dense(&x, n, 1, &positive);
    let binary = labeled_dense(&x, n, 1, &binary);
    let ranked = labeled_dense(&x, n, 1, &counts)
        .with_group_sizes(&[20, 20])
        .unwrap();
    let fit = |builder: crate::config::TrainingParamsBuilder, d: &DMatrix| {
        train(&builder.max_depth(2).build().unwrap(), d, 2).unwrap()
    };
    let b = TrainingParams::builder;
    let ranker = || b().objective(Objective::RankNdcg(LambdaRank::new(5).unwrap()));
    // (configuration, data, block, key, exported text, retained value)
    let cases: [(_, _, _, _, _, Retained); 7] = [
        (
            b().objective(Objective::Tweedie(Tweedie::new(1.2).unwrap())),
            &d,
            "tweedie_regression_param",
            "tweedie_variance_power",
            "1.2",
            |m| match m.objective().built_in() {
                Some(Objective::Tweedie(t)) => Some(t.variance_power()),
                _ => None,
            },
        ),
        (
            b().objective(Objective::Poisson)
                .max_delta_step(MaxDeltaStep::Bounded(0.3)),
            &d,
            "poisson_regression_param",
            "max_delta_step",
            "0.3",
            |m| Some(m.max_delta_step()),
        ),
        (
            b().objective(Objective::PseudoHuber(PseudoHuber::new(2.5).unwrap())),
            &d,
            "pseudo_huber_param",
            "huber_slope",
            "2.5",
            |m| match m.objective().built_in() {
                Some(Objective::PseudoHuber(h)) => Some(h.slope()),
                _ => None,
            },
        ),
        (
            b().objective(Objective::BinaryLogistic(RegLoss::new(3.0).unwrap())),
            &binary,
            "reg_loss_param",
            "scale_pos_weight",
            "3",
            |m| match m.objective().built_in() {
                Some(Objective::BinaryLogistic(l)) => Some(l.scale_pos_weight()),
                _ => None,
            },
        ),
        (
            b().objective(Objective::SquaredError(RegLoss::new(0.5).unwrap())),
            &d,
            "reg_loss_param",
            "scale_pos_weight",
            "0.5",
            |m| match m.objective().built_in() {
                Some(Objective::SquaredError(r)) => Some(r.scale_pos_weight()),
                _ => None,
            },
        ),
        (
            b().objective(Objective::Gamma(RegLoss::new(2.0).unwrap())),
            &positive,
            "reg_loss_param",
            "scale_pos_weight",
            "2",
            |m| match m.objective().built_in() {
                Some(Objective::Gamma(r)) => Some(r.scale_pos_weight()),
                _ => None,
            },
        ),
        (
            ranker(),
            &ranked,
            "lambdarank_param",
            "lambdarank_num_pair_per_sample",
            "5",
            |m| match m.objective().built_in() {
                Some(Objective::RankNdcg(r)) => Some(r.num_pair_per_sample() as f64),
                _ => None,
            },
        ),
    ];
    for (builder, data, block, key, text, retained) in cases {
        let (exported, json) = export_json_document(&fit(builder, data));
        assert_eq!(
            json["learner"]["objective"][block][key], text,
            "{block}.{key}"
        );
        let back = import_xgboost_json(&exported).unwrap();
        let expected: f64 = text.parse().unwrap();
        assert_eq!(retained(&back), Some(expected), "{block}.{key}");
    }

    // XGBoost's own "not set" sentinel maps to the `topk` default.
    let exported = export_xgboost_json(&fit(ranker(), &ranked))
        .unwrap()
        .replace(
            r#""lambdarank_num_pair_per_sample": "5""#,
            r#""lambdarank_num_pair_per_sample": "4294967295""#,
        );
    let unset = import_xgboost_json(&exported).unwrap();
    assert_eq!(
        unset.objective().built_in(),
        Some(&Objective::RankNdcg(LambdaRank::default()))
    );
}

/// Alpha lists travel as XGBoost's array strings (either bracket form),
/// `num_target` counts the per-alpha outputs, and a list that does not
/// rebuild the objective is a format error, never an untransformed model.
#[test]
fn alpha_list_objectives_roundtrip_and_reject_bad_blocks() {
    let n = 40;
    let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
    let y: Vec<f32> = x.iter().map(|&v| 3.0 * v + (v * 17.0).sin()).collect();
    let d = labeled_dense(&x, n, 1, &y);
    let params = TrainingParams::builder()
        .objective(Objective::Quantile(Quantiles::new(vec![0.1, 0.9]).unwrap()))
        .max_depth(2)
        .build()
        .unwrap();
    let model = train(&params, &d, 3).unwrap();
    let (exported, json) = export_json_document(&model);
    assert_eq!(
        json["learner"]["objective"],
        json!({"name": "reg:quantileerror", "quantile_loss_param": {"quantile_alpha": "[0.1,0.9]"}})
    );
    assert_eq!(json["learner"]["learner_model_param"]["num_target"], "2");
    let back = import_xgboost_json(&exported).unwrap();
    assert_eq!(back.n_outputs(), 2);
    assert_eq!(back.n_targets(), 1);
    assert_eq!(back.predict(&d).unwrap(), model.predict(&d).unwrap());

    let parenthesized = exported.replace("[0.1,0.9]", "(0.1, 0.9)");
    let back = import_xgboost_json(&parenthesized).unwrap();
    assert_eq!(back.predict(&d).unwrap(), model.predict(&d).unwrap());
    for bad in ["0.5", "[0.9,0.1]", "[]", "nope"] {
        assert_format_error(
            import_xgboost_json(&exported.replace("[0.1,0.9]", bad)),
            bad,
        );
    }

    let mae = TrainingParams::builder()
        .objective(Objective::AbsoluteError)
        .max_depth(2)
        .build()
        .unwrap();
    let (_, json) = export_json_document(&train(&mae, &d, 2).unwrap());
    assert_eq!(
        json["learner"]["objective"],
        json!({"name": "reg:absoluteerror"})
    );
}

/// `survival:aft` keeps its distribution and scale in `aft_loss_param`
/// and `survival:cox` writes no parameter block; both store `base_score`
/// as `exp(margin)` and predict identically after the round trip.
#[test]
fn survival_objectives_roundtrip() {
    let n = 40;
    let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
    let times: Vec<f32> = (0..n).map(|i| 1.0 + (i % 7) as f32).collect();
    let upper: Vec<f32> = times
        .iter()
        .enumerate()
        .map(|(i, &t)| if i % 3 == 0 { f32::INFINITY } else { t })
        .collect();
    let d = DMatrix::from_dense(&x, n, 1)
        .unwrap()
        .with_label_bounds(&times, &upper)
        .unwrap();
    let params = TrainingParams::builder()
        .objective(Objective::Aft(
            Aft::new(AftDistribution::Extreme, 1.5).unwrap(),
        ))
        .max_depth(2)
        .build()
        .unwrap();
    let aft = train(&params, &d, 3).unwrap();
    let (exported, json) = export_json_document(&aft);
    assert_eq!(
        json["learner"]["objective"],
        json!({"name": "survival:aft", "aft_loss_param": {
            "aft_loss_distribution": "extreme", "aft_loss_distribution_scale": "1.5"}})
    );
    assert_eq!(
        json["learner"]["learner_model_param"]["base_score"],
        "[0.5]"
    );
    let back = import_xgboost_json(&exported).unwrap();
    assert_eq!(
        back.objective().built_in(),
        Some(&Objective::Aft(
            Aft::new(AftDistribution::Extreme, 1.5).unwrap()
        ))
    );
    assert_eq!(back.predict(&d).unwrap(), aft.predict(&d).unwrap());

    let signed: Vec<f32> = times
        .iter()
        .zip(&upper)
        .map(|(&t, &u)| if u.is_infinite() { -t } else { t })
        .collect();
    let dc = labeled_dense(&x, n, 1, &signed);
    let params = TrainingParams::builder()
        .objective(Objective::Cox)
        .max_depth(2)
        .build()
        .unwrap();
    let cox = train(&params, &dc, 3).unwrap();
    let (exported, json) = export_json_document(&cox);
    assert_eq!(
        json["learner"]["objective"],
        json!({"name": "survival:cox"})
    );
    let back = import_xgboost_json(&exported).unwrap();
    assert_eq!(back.base_scores(), cox.base_scores());
    assert_eq!(back.predict(&dc).unwrap(), cox.predict(&dc).unwrap());
}

/// An XE-NDCG model (a hessboost extension) round-trips through the
/// native format, but XGBoost's model format cannot carry it.
#[test]
fn xendcg_native_roundtrip_and_xgboost_exports_refused() {
    let data = labeled_dense(&[0.0, 1.0, 2.0, 0.5], 4, 1, &[0.0, 1.0, 2.0, 1.0])
        .with_group_sizes(&[2, 2])
        .unwrap();
    let params = TrainingParams::builder()
        .objective(Objective::RankXendcg)
        .max_depth(2)
        .seed(9)
        .build()
        .unwrap();
    let model = train(&params, &data, 3).unwrap();
    let bytes = model.to_bytes().unwrap();
    let restored = BoostedModel::from_bytes(&bytes).unwrap();
    assert_eq!(restored.to_bytes().unwrap(), bytes);
    assert_eq!(restored.objective(), model.objective());
    assert_eq!(
        restored.predict(&data).unwrap(),
        model.predict(&data).unwrap()
    );
    assert_format_error(export_xgboost_json(&model), "hessboost extension");
    assert_format_error(export_xgboost_ubjson(&model), "hessboost extension");
}

#[test]
fn custom_objective_export_is_rejected() {
    use crate::objective::{CustomLoss, GradPair};
    let (_, d) = reg_model();
    let obj = CustomLoss::new("custom:test", 1, |preds, labels, w, out| {
        for i in 0..preds.len() {
            let wi = w.map_or(1.0, |ws| ws[i]);
            out[i] = GradPair::new((preds[i] - labels[i]) * wi, wi);
        }
    });
    let params = TrainingParams::builder()
        .objective(Objective::custom(obj))
        .max_depth(2)
        .build()
        .unwrap();
    let model = train(&params, &d, 2).unwrap();
    assert_format_error(export_xgboost_json(&model), "custom objective");
}
