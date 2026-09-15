//! Split-selection parity regression tests.
//!
//! The dense vector split scan is a prefilter whose accepted candidate must
//! be the one the sequential scalar scan would pick for the same node. That
//! holds only when the scan continues the node-wide gain-epsilon incumbent
//! instead of starting a fresh per-feature one: with gains chosen inside the
//! 1e-6 epsilon band, a fresh incumbent accepts a candidate the node-wide
//! sequence rejects, discards the one it accepts, and silently moves the
//! split to another feature.

use sequoia_boost::prelude::*;

/// One boosting round on a fixed gradient table, so the root split depends
/// only on the split scan.
#[test]
fn vector_split_scan_continues_the_node_incumbent() {
    // Four real rows carry unit-Hessian gradients chosen so that with
    // lambda = alpha = 0 and min_child_weight = 1, the best split of
    // feature 0 gains about 1.92e-6, while feature 1 has two candidates at
    // about 2.61e-6 and 3.24e-6. The scalar scan rejects the first feature-1
    // candidate (inside the 1e-6 epsilon of the incumbent) and accepts the
    // second, so the root must split on feature 1. A per-feature incumbent
    // instead accepts the weaker candidate, discards the stronger one, and
    // leaves the root split on feature 0.
    const REAL: usize = 4;
    const PADS: usize = 16;
    let rows = REAL + PADS;
    let gradients = [0.0014f32, 0.0004, 0.0012, -0.003];

    // Rows 4.. are weight-zero padding: their distinct values give each
    // feature more than the 16 bins the vector scan requires, and their zero
    // Hessian keeps every real candidate's statistics unchanged.
    let mut features = vec![0f32; rows * 2];
    for pad in 0..PADS {
        let value = pad as f32 * 0.01;
        features[(REAL + pad) * 2] = value;
        features[(REAL + pad) * 2 + 1] = value;
    }
    // Feature 0 bins: c alone, then a, b, d together.
    features[0] = 0.40; // a
    features[2] = 0.40; // b
    features[4] = 0.30; // c
    features[6] = 0.40; // d
                        // Feature 1 bins: a, then b, then c and d together.
    features[1] = 0.30; // a
    features[3] = 0.35; // b
    features[7] = 0.40; // d
    features[5] = 0.40; // c

    let labels = vec![0f32; rows];
    let dtrain = DMatrix::from_dense(&features, rows, 2)
        .unwrap()
        .with_labels(&labels)
        .unwrap();
    let objective = CustomObjective::new("reg:fixture", 1, 0.0, "rmse", move |_, _, _, out| {
        for i in 0..REAL {
            out[i] = GradPair::new(gradients[i], 1.0);
        }
        for out in &mut out[REAL..] {
            *out = GradPair::new(0.0, 0.0);
        }
    });

    let params = TrainingParams::builder()
        .tree_method(TreeMethod::Hist)
        .max_depth(1)
        .eta(1.0)
        .max_bin(256)
        .lambda(0.0)
        .alpha(0.0)
        .min_child_weight(1.0)
        .build()
        .unwrap();
    let model = train_with_objective(&params, &dtrain, 1, Box::new(objective)).unwrap();

    // The scalar scan splits the root on feature 1 at the cut 0.40. Both
    // probe rows sit above every feature-0 value, so they share a leaf under
    // a feature-0 root split and land on opposite leaves under the correct
    // feature-1 split: left -(a + b)/2 = -9e-4, right -(c + d)/2 = 9e-4.
    let probe = DMatrix::from_dense(&[0.45, 0.34, 0.45, 0.41], 2, 2)
        .unwrap()
        .with_labels(&[0f32, 0.0])
        .unwrap();
    let preds = model.predict(&probe).unwrap();
    assert!(
        (preds[0] - -0.0009).abs() < 1e-6,
        "left leaf {} should be -(a+b)/2",
        preds[0]
    );
    assert!(
        (preds[1] - 0.0009).abs() < 1e-6,
        "right leaf {} should be -(c+d)/2",
        preds[1]
    );
}
