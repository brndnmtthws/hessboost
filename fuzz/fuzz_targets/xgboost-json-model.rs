#![no_main]
//! The XGBoost JSON model importer (`BoostedModel::from_xgboost_json`):
//! arbitrary text either fails to import or yields a model every prediction
//! and serialization API handles.
use hessboost::prelude::*;

#[path = "common.rs"]
mod common;

crate::text_target!(BoostedModel::from_xgboost_json);
