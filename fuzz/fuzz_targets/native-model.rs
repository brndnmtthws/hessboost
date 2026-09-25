#![no_main]
//! The native binary model decoder (`BoostedModel::from_bytes`): arbitrary
//! bytes either fail to decode or yield a model every prediction and
//! serialization API handles.
//!
//! A leading `0` byte passes the rest through unchanged (zstd frames,
//! headers, checksums). Otherwise the rest is a section table that the
//! target seals in an uncompressed container with a valid checksum, so
//! mutations reach the section and model decoders instead of dying at the
//! checksum.
use hessboost::prelude::*;
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;

/// `b"HBM\0"` and the container version (`model/native.rs`).
const HEADER: &[u8] = b"HBM\0\x03";

fuzz_target!(|data: &[u8]| {
    let Some((&mode, rest)) = data.split_first() else {
        return;
    };
    let container = if mode == 0 {
        rest.to_vec()
    } else {
        let mut container = [HEADER, rest].concat();
        let checksum = xxhash_rust::xxh64::xxh64(&container, 0);
        container.extend_from_slice(&checksum.to_le_bytes());
        container
    };
    if let Ok(model) = BoostedModel::from_bytes(&container) {
        common::exercise(&model);
    }
});
