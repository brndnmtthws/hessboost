#![no_main]
//! The XGBoost UBJSON model importer (`ModelFormat::XgboostUbjson`):
//! arbitrary bytes either fail to import or yield a model every prediction
//! and serialization API handles.
use hessboost::prelude::*;

#[path = "common.rs"]
mod common;

crate::bytes_target!(ModelFormat::XgboostUbjson);
