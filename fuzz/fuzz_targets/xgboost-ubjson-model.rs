#![no_main]
//! The XGBoost UBJSON model importer (`BoostedModel::from_xgboost_ubjson`):
//! arbitrary bytes either fail to import or yield a model every prediction
//! and serialization API handles.
use hessboost::prelude::*;
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;

fuzz_target!(|data: &[u8]| {
    if let Ok(model) = BoostedModel::from_xgboost_ubjson(data) {
        common::exercise(&model);
    }
});
