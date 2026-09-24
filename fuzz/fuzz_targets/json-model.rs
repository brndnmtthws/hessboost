#![no_main]
//! The native JSON model parser (`BoostedModel::from_json`): arbitrary text
//! either fails to parse or yields a model every prediction and
//! serialization API handles.
use hessboost::prelude::*;
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    if let Ok(model) = BoostedModel::from_json(text) {
        common::exercise(&model);
    }
});
