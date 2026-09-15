//! Regression tests for SHAP contribution accumulation across trees.

use sequoia_boost::prelude::*;

/// A later constant tree must not round away an earlier tree's contribution.
///
/// Tree 1 gives feature 0 a contribution of +2 on the right branch. Tree 2 is
/// constant with leaf values of 2^60, so its own per-tree contribution is
/// exactly zero: its hot and cold elements cancel in f64. Accumulating trees
/// straight into the row buffer instead adds 2^59 and subtracts it around the
/// earlier +2, and the f64 rounding deletes the +2.
#[test]
fn contributions_survive_a_later_constant_tree() {
    let huge = (1u64 << 60) as f32;
    let model_json = format!(
        r#"{{
  "trees": [
    {{
      "nodes": [
        {{"split_feature": 0, "split_cond": 0.5, "default_left": false, "left": 1, "right": 2, "leaf_value": 0.0, "sum_hess": 2.0, "split_gain": 1.0, "is_categorical": false, "cat_begin": 0, "cat_end": 0}},
        {{"split_feature": 0, "split_cond": 0.0, "default_left": true, "left": -1, "right": -1, "leaf_value": -2.0, "sum_hess": 1.0, "split_gain": 0.0, "is_categorical": false, "cat_begin": 0, "cat_end": 0}},
        {{"split_feature": 0, "split_cond": 0.0, "default_left": true, "left": -1, "right": -1, "leaf_value": 2.0, "sum_hess": 1.0, "split_gain": 0.0, "is_categorical": false, "cat_begin": 0, "cat_end": 0}}
      ]
    }},
    {{
      "nodes": [
        {{"split_feature": 0, "split_cond": 0.5, "default_left": false, "left": 1, "right": 2, "leaf_value": 0.0, "sum_hess": 2.0, "split_gain": 0.0, "is_categorical": false, "cat_begin": 0, "cat_end": 0}},
        {{"split_feature": 0, "split_cond": 0.0, "default_left": true, "left": -1, "right": -1, "leaf_value": {huge}, "sum_hess": 1.0, "split_gain": 0.0, "is_categorical": false, "cat_begin": 0, "cat_end": 0}},
        {{"split_feature": 0, "split_cond": 0.0, "default_left": true, "left": -1, "right": -1, "leaf_value": {huge}, "sum_hess": 1.0, "split_gain": 0.0, "is_categorical": false, "cat_begin": 0, "cat_end": 0}}
      ]
    }}
  ],
  "base_score": 0.0,
  "objective": "reg:squarederror",
  "num_class": 1,
  "n_features": 1
}}"#
    );
    let model = BoostedModel::from_json(&model_json).unwrap();
    let row = DMatrix::from_dense(&[0.7f32], 1, 1)
        .unwrap()
        .with_labels(&[0f32])
        .unwrap();

    // One row of [feature 0, bias].
    let contribs = model.predict_contribs(&row).unwrap();
    assert_eq!(contribs.len(), 2);
    assert_eq!(contribs[0], 2.0, "tree 1's contribution must survive");

    // One row of (n_features + 1)^2 = 4 values; the diagonal entry for
    // feature 0 is its main effect and must match the contribution.
    let interactions = model.predict_interactions(&row).unwrap();
    assert_eq!(interactions.len(), 4);
    assert_eq!(
        interactions[0], 2.0,
        "the interaction diagonal must match contributions"
    );
}
