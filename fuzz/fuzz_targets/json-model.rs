#![no_main]
//! The native JSON model parser (`BoostedModel::from_json`): arbitrary text
//! either fails to parse or yields a model every prediction and
//! serialization API handles.
use hessboost::prelude::*;

#[path = "common.rs"]
mod common;

crate::text_target!(BoostedModel::from_json);
