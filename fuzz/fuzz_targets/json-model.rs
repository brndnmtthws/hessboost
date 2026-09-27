#![no_main]
//! The native JSON model parser (`ModelFormat::Json`): arbitrary text
//! either fails to parse or yields a model every prediction and
//! serialization API handles.
use hessboost::prelude::*;

#[path = "common.rs"]
mod common;

crate::text_target!(ModelFormat::Json);
