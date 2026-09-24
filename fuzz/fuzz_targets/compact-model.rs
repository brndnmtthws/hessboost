#![no_main]
//! The compact (`HBTD`) model parser (`CompactModel::from_bytes`): arbitrary
//! bytes either fail to parse or yield a model that predicts without
//! panicking, with the documented output shapes.
use hessboost::model::compact::CompactModel;
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;

/// Widest model the target predicts with (see `common.rs`).
const MAX_PREDICT_FEATURES: usize = 64;
const MAX_OUTPUTS: usize = 16;

fuzz_target!(|data: &[u8]| {
    let Ok(model) = CompactModel::from_bytes(data) else {
        return;
    };
    assert!(
        model.to_bytes() == data,
        "to_bytes returns the parsed bytes"
    );
    let n_features = model.n_features();
    let k = model.n_outputs();
    assert!(n_features > 0 && k > 0);
    for feature in model.used_features() {
        assert!(feature < n_features);
    }
    if n_features > MAX_PREDICT_FEATURES || k > MAX_OUTPUTS {
        return;
    }
    let probe = common::probe_matrix(n_features);
    let margin = model
        .predict_margin(&probe)
        .expect("probe matrix matches the model");
    assert_eq!(margin.len(), probe.n_rows() * k);
    let preds = model
        .predict(&probe)
        .expect("probe matrix matches the model");
    let expected = if model.objective() == "multi:softmax" {
        1
    } else {
        k
    };
    assert_eq!(preds.len(), probe.n_rows() * expected);
});
