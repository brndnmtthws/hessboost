#![no_main]
//! The XGBoost UBJSON model importer (`BoostedModel::from_xgboost_ubjson`):
//! arbitrary bytes either fail to import or yield a model every prediction
//! and serialization API handles.
use hessboost::prelude::*;

#[path = "common.rs"]
mod common;

crate::bytes_target!(BoostedModel::from_xgboost_ubjson);
