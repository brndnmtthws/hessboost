#![no_main]
//! The LightGBM text model importer (`ModelFormat::LightgbmText`):
//! arbitrary text either fails to import or yields a model every prediction
//! and serialization API handles.
use hessboost::prelude::*;

#[path = "common.rs"]
mod common;

crate::text_target!(ModelFormat::LightgbmText);
