#![no_main]
//! The compact (`HBTD`) model parser (`CompactModel::from_bytes`): arbitrary
//! bytes either fail to parse or yield a model that predicts without
//! panicking, with the documented output shapes.
use hessboost::model::compact::CompactModel;
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;


fuzz_target!(|data: &[u8]| {
    let Ok(model) = CompactModel::from_bytes(data) else {
        return;
    };
    assert!(
        model.to_bytes() == data,
        "to_bytes returns the parsed bytes"
    );
    common::exercise_compact(&model);
});
