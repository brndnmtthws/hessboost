//! Regression tests for SHAP contribution accumulation across trees.

use hessboost::prelude::*;

/// A later constant tree must not round away an earlier tree's contribution.
///
/// Tree 1 gives feature 0 a contribution of +2 on the right branch. Tree 2 is
/// constant with leaf values of 2^60, so its own per-tree contribution is
/// exactly zero: its two return edges cancel within the tree's buffer.
/// Accumulating trees straight into the row buffer instead adds ±2^59 around
/// the earlier +2, and the rounding deletes the +2. XGBoost 3.4.2 buffers
/// per tree for `pred_contribs` (but not for `pred_interactions`, whose
/// diagonal loses the +2 in both implementations).
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
  "base_score": [0.0],
  "objective": "reg:squarederror",
  "num_class": 0,
  "n_outputs": 1,
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
}

/// Coverless splits (e.g. refreshed or hand-built models) follow each branch
/// with probability 1/2 and their expected value averages the children, so
/// the attributions stay additive. Expected values are XGBoost 3.4.2's
/// `pred_contribs` / `pred_interactions` on the same model.
#[test]
fn zero_cover_splits_average_their_children() {
    let node = |feature: u32, left: i32, right: i32, value: f32| {
        format!(
            r#"{{"split_feature": {feature}, "split_cond": 0.5, "default_left": true, "left": {left}, "right": {right}, "leaf_value": {value}, "sum_hess": 0.0, "split_gain": 0.0, "is_categorical": false, "cat_begin": 0, "cat_end": 0}}"#
        )
    };
    // x0 < 0.5 ? -1 : (x1 < 0.5 ? 2 : 4), every node with zero cover.
    let model_json = format!(
        r#"{{"trees": [{{"nodes": [{}, {}, {}, {}, {}]}}], "base_score": [0.0], "objective": "reg:squarederror", "num_class": 0, "n_outputs": 1, "n_features": 2}}"#,
        node(0, 1, 2, 0.0),
        node(0, -1, -1, -1.0),
        node(1, 3, 4, 0.0),
        node(0, -1, -1, 2.0),
        node(0, -1, -1, 4.0),
    );
    let model = BoostedModel::from_json(&model_json).unwrap();
    let row = DMatrix::from_dense(&[0.7f32, 0.2], 1, 2).unwrap();

    // E[f] = 0.5 * -1 + 0.25 * 2 + 0.25 * 4 = 1; margin 2.
    let contribs = model.predict_contribs(&row).unwrap();
    assert_eq!(contribs, [1.75, -0.75, 1.0]);
    let interactions = model.predict_interactions(&row).unwrap();
    let want = [2.0, -0.25, 0.0, -0.25, -0.5, 0.0, 0.0, 0.0, 1.0];
    for (got, want) in interactions.iter().zip(want) {
        assert!((got - want).abs() < 1e-6, "{interactions:?}");
    }
}
