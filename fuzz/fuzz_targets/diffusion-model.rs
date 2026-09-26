#![no_main]
//! The diffusion model decoders (`DiffusionModel::from_bytes` and
//! `from_json`): arbitrary input either fails to decode or yields a model
//! that samples and round-trips.
//!
//! The first byte picks the decoder: `0` passes the rest to `from_bytes`
//! unchanged; `1` seals the rest as a section table in an uncompressed
//! `HBDM` container with a valid checksum, so mutations reach the section
//! decoder; anything else parses the rest as JSON.
use hessboost::diffusion::DiffusionModel;
use hessboost::prelude::*;
use libfuzzer_sys::fuzz_target;

/// `b"HBDM"` and the container version (`diffusion/format.rs`).
const HEADER: &[u8] = b"HBDM\x01";
/// Largest model the target samples from: a fuzzed header can claim any
/// step or feature count, which only makes sampling slow.
const MAX_STEPS: usize = 64;
const MAX_FEATURES: usize = 64;
const MAX_TREES: usize = 4096;

fuzz_target!(|data: &[u8]| {
    let Some((&mode, rest)) = data.split_first() else {
        return;
    };
    let model = match mode {
        0 => DiffusionModel::from_bytes(rest),
        1 => {
            let mut container = [HEADER, rest].concat();
            let checksum = xxhash_rust::xxh64::xxh64(&container, 0);
            container.extend_from_slice(&checksum.to_le_bytes());
            DiffusionModel::from_bytes(&container)
        }
        _ => match std::str::from_utf8(rest) {
            Ok(text) => DiffusionModel::from_json(text),
            Err(_) => return,
        },
    };
    let Ok(model) = model else {
        return;
    };
    let from_bytes =
        DiffusionModel::from_bytes(&model.to_bytes().expect("an accepted model saves"))
            .expect("a saved model loads");
    let from_json = DiffusionModel::from_json(&model.to_json().expect("an accepted model saves"))
        .expect("a saved model loads");
    assert_eq!(from_bytes.method(), model.method());
    assert_eq!(from_json.method(), model.method());
    if model.n_steps() <= MAX_STEPS
        && model.n_features() <= MAX_FEATURES
        && model.n_outputs() <= MAX_FEATURES
        && model.regressor().num_trees() <= MAX_TREES
    {
        let probe = DMatrix::from_dense(&vec![0.0; 2 * model.n_features()], 2, model.n_features())
            .expect("probe matrix is valid");
        // Sampling may refuse a diverging sampler, but never panics, and a
        // successful draw has the documented shape and reloads identically.
        if let Ok(samples) = model.sample(&probe, 2, 0) {
            assert_eq!(samples.values().len(), 2 * 2 * model.n_outputs());
            assert_eq!(from_bytes.sample(&probe, 2, 0).ok(), Some(samples));
        }
    }
});
