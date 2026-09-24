//! SHAP contribution accumulation across trees.

use hessboost::config::ObjectiveParams;
use hessboost::prelude::*;

/// One node of a hand-built native-JSON tree: an `x[feature] < 0.5` split
/// (missing values go left), or a leaf when `left == right == -1`.
fn node(feature: u32, left: i32, right: i32, value: f64, hess: f64) -> String {
    format!(
        r#"{{"split_feature": {feature}, "split_cond": 0.5, "default_left": true, "left": {left}, "right": {right}, "leaf_value": {value:e}, "sum_hess": {hess:e}, "split_gain": 0.0, "is_categorical": false, "cat_begin": 0, "cat_end": 0}}"#
    )
}

/// The native-JSON `objective_params` object of a model without
/// objective-specific parameters.
fn default_objective_params() -> String {
    serde_json::to_string(&ObjectiveParams::default()).unwrap()
}

/// A single-output squared-error model over `n_features` whose trees are
/// the given [`node`] lists.
fn scalar_model(trees: &[&[String]], n_features: usize) -> BoostedModel {
    let trees: Vec<String> = trees
        .iter()
        .map(|nodes| {
            format!(
                r#"{{"nodes": [{}], "categories": [], "size_leaf_vector": 0, "leaf_vectors": []}}"#,
                nodes.join(", ")
            )
        })
        .collect();
    BoostedModel::from_json(&format!(
        r#"{{"trees": [{}], "tree_weights": [], "base_score": [0.0], "objective": "reg:squarederror", "objective_params": {}, "num_class": 0, "n_outputs": 1, "n_targets": 1, "num_parallel_tree": 1, "n_features": {n_features}}}"#,
        trees.join(", "),
        default_objective_params()
    ))
    .unwrap()
}

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
    let huge = 2f64.powi(60);
    let model = scalar_model(
        &[
            &[
                node(0, 1, 2, 0.0, 2.0),
                node(0, -1, -1, -2.0, 1.0),
                node(0, -1, -1, 2.0, 1.0),
            ],
            &[
                node(0, 1, 2, 0.0, 2.0),
                node(0, -1, -1, huge, 1.0),
                node(0, -1, -1, huge, 1.0),
            ],
        ],
        1,
    );
    let row = DMatrix::from_dense(&[0.7f32], 1, 1).unwrap();

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
    // x0 < 0.5 ? -1 : (x1 < 0.5 ? 2 : 4), every node with zero cover.
    let model = scalar_model(
        &[&[
            node(0, 1, 2, 0.0, 0.0),
            node(0, -1, -1, -1.0, 0.0),
            node(1, 3, 4, 0.0, 0.0),
            node(0, -1, -1, 2.0, 0.0),
            node(0, -1, -1, 4.0, 0.0),
        ]],
        2,
    );
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

/// A feature repeated down a path overwrites its probability: the basis is
/// multiplied by the new factor and divided by the old one. With hot-child
/// covers `1 → 2^-32 → 2^-64 → 2^-96 → 2^-128` the third split's basis
/// product exceeds `f32` if formed before the division, and the fourth
/// split's path probability `2^128` exceeds it outright, although the
/// resulting basis and attributions are representable; every attribution
/// must stay finite.
#[test]
fn repeated_feature_basis_update_stays_finite() {
    let cover = |e: i32| 2f64.powi(-e);
    // x0 < 0.5 four times reaches the only non-zero leaf.
    let model = scalar_model(
        &[&[
            node(0, 1, 2, 0.0, cover(0)),
            node(0, 3, 4, 0.0, cover(32)),
            node(0, -1, -1, 0.0, cover(0)),
            node(0, 5, 6, 0.0, cover(64)),
            node(0, -1, -1, 0.0, cover(32)),
            node(0, 7, 8, 0.0, cover(96)),
            node(0, -1, -1, 0.0, cover(64)),
            node(0, -1, -1, 1.0, cover(128)),
            node(0, -1, -1, 0.0, cover(96)),
        ]],
        1,
    );
    let row = DMatrix::from_dense(&[0.2f32], 1, 1).unwrap();
    assert_eq!(model.predict_margin(&row).unwrap(), [1.0]);

    // E[f] = 2^-128, so feature 0 carries the whole margin.
    let bias = cover(128) as f32;
    assert!(bias > 0.0);
    let contribs = model.predict_contribs(&row).unwrap();
    assert_eq!(contribs[1], bias, "{contribs:?}");
    assert!((contribs[0] - 1.0).abs() < 1e-6, "{contribs:?}");
    let interactions = model.predict_interactions(&row).unwrap();
    assert_eq!(interactions[3], bias, "{interactions:?}");
    assert!((interactions[0] - 1.0).abs() < 1e-6, "{interactions:?}");
    assert_eq!(interactions[1..3], [0.0, 0.0], "{interactions:?}");
}

/// XGBoost 3.4.2 takes a vector-leaf tree's expected values top-down
/// (`FillRootMeanValues`): each leaf vector enters scaled by its path's cover
/// fraction, in leaf order. For output 0, `x0 < 0.5 ? 2^60 : (x1 < 0.5 ?
/// -2^60 : 1)` with covers 3 / 1, 2 / 1, 1 that sums `2^60/3 - 2^60/3 + 1/3`,
/// so the bias is `1/3`; a per-output bottom-up reduction loses the `1` to
/// `-2^60` in the right subtree and reports `0`.
#[test]
fn vector_leaf_bias_accumulates_leaves_top_down() {
    let huge = 2f64.powi(60);
    let model_json = format!(
        r#"{{"trees": [{{"nodes": [{}, {}, {}, {}, {}], "categories": [], "size_leaf_vector": 2, "leaf_vectors": [0.0, 0.0, {huge:e}, 0.0, 0.0, 0.0, {neg:e}, 0.0, 1.0, 0.0]}}], "tree_weights": [], "base_score": [0.0, 0.0], "objective": "reg:squarederror", "objective_params": {objective_params}, "num_class": 0, "n_outputs": 2, "n_targets": 2, "num_parallel_tree": 1, "n_features": 2}}"#,
        node(0, 1, 2, 0.0, 3.0),
        node(0, -1, -1, 0.0, 1.0),
        node(1, 3, 4, 0.0, 2.0),
        node(0, -1, -1, 0.0, 1.0),
        node(0, -1, -1, 0.0, 1.0),
        neg = -huge,
        objective_params = default_objective_params(),
    );
    let model = BoostedModel::from_json(&model_json).unwrap();
    let row = DMatrix::from_dense(&[0.7f32, 0.7], 1, 2).unwrap();
    assert_eq!(model.predict_margin(&row).unwrap(), [1.0, 0.0]);

    let third = (1.0f64 / 3.0) as f32;
    // Per output: [x0, x1, bias].
    let contribs = model.predict_contribs(&row).unwrap();
    assert_eq!([contribs[2], contribs[5]], [third, 0.0], "{contribs:?}");
    // Per output: a 3 × 3 matrix with the bias in its last cell.
    let interactions = model.predict_interactions(&row).unwrap();
    assert_eq!(
        [interactions[8], interactions[17]],
        [third, 0.0],
        "{interactions:?}"
    );
}
