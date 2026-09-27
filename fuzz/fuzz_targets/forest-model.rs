#![no_main]
//! The forest model decoders (`ForestModel::from_bytes` and `from_json`):
//! arbitrary input either fails to decode or yields a model that generates,
//! imputes, and round-trips.
//!
//! The first byte picks the decoder: `0` passes the rest to `from_bytes`
//! unchanged; `1` seals the rest as a section table in an uncompressed
//! `HBFF` container with a valid checksum, so mutations reach the section
//! decoder; anything else parses the rest as JSON.
use hessboost::diffusion::forest::{ForestMethod, ForestModel};
use hessboost::prelude::*;
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;

/// `b"HBFF"` and the container version (`diffusion/forest/format.rs`).
const HEADER: &[u8] = b"HBFF\x01";
/// Largest model the target samples from: a fuzzed header can claim any
/// level or column count, which only makes sampling slow.
const MAX_LEVELS: usize = 64;
const MAX_COLUMNS: usize = 64;
const MAX_TREES: usize = 4096;

fuzz_target!(|data: &[u8]| {
    let Some((&mode, rest)) = data.split_first() else {
        return;
    };
    let Some(model) = common::parse_mode(
        mode,
        rest,
        HEADER,
        ForestModel::from_bytes,
        ForestModel::from_json,
    ) else {
        return;
    };
    let Ok(model) = model else {
        return;
    };
    common::round_trip(
        &model,
        |model| model.to_bytes(),
        ForestModel::from_bytes,
        |model| model.to_json(),
        ForestModel::from_json,
        |model, from_bytes| assert_eq!(from_bytes.method(), model.method()),
    );
    let from_bytes = ForestModel::from_bytes(&model.to_bytes().expect("an accepted model saves"))
        .expect("a saved model loads");
    assert_eq!(from_bytes.classes(), model.classes());
    let trees: usize = model.gbdts().iter().map(BoostedModel::num_trees).sum();
    if model.n_t() > MAX_LEVELS || model.n_columns() > MAX_COLUMNS || trees > MAX_TREES {
        return;
    }
    // Generation may refuse a diverging sampler, but never panics, and a
    // successful draw has the documented shape and reloads identically.
    if let Ok(rows) = model.generate(2, 0) {
        assert_eq!(rows.as_slice().len(), 2 * model.n_columns());
        assert!(rows.as_slice().iter().all(|v| v.is_finite()));
        assert_eq!(from_bytes.generate(2, 0).ok(), Some(rows));
    }
    if matches!(model.method(), ForestMethod::Diffusion { .. }) {
        let mut row = vec![0.0f32; model.n_columns()];
        row[0] = f32::NAN;
        let mut probe = DMatrix::from_dense(&row, 1, model.n_columns()).expect("probe is valid");
        if let Some(&class) = model.classes().first() {
            probe = probe.with_labels(&[class as f32]).expect("one label");
        }
        // Zero may be an unseen category, which is refused; otherwise the
        // missing entry is filled.
        if let Ok(imputed) = model.impute(&probe, 1, None, 0) {
            // One imputation of one row: `[imputation][row][column]`.
            assert_eq!(
                (
                    imputed.n_imputations(),
                    imputed.n_rows(),
                    imputed.n_columns()
                ),
                (1, 1, model.n_columns())
            );
            assert_eq!(imputed.as_slice().len(), model.n_columns());
            assert!(
                imputed
                    .get(0, 0)
                    .is_some_and(|r| r.iter().all(|v| v.is_finite()))
            );
        }
    }
});
