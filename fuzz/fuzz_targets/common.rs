//! Model checks shared by the fuzz targets. Each target is its own `[[bin]]`,
//! so this file is pulled in per target with `#[path]`.
#![allow(dead_code, reason = "each target uses a different subset")]

use hessboost::model::compact::CompactModel;
use hessboost::model::{ImportanceType, Predictions};
use hessboost::objective::distributional::DistFamily;
use hessboost::prelude::*;
use xxhash_rust::xxh64::xxh64;

/// Seal fuzzed container bytes with a format header and valid xxh64 checksum.
pub fn seal(header: &[u8], rest: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(header.len() + rest.len() + 8);
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(rest);
    let checksum = xxh64(&bytes, 0);
    bytes.extend_from_slice(&checksum.to_le_bytes());
    bytes
}

/// Decode the three shared mode-byte choices: raw bytes, sealed container, or JSON text.
pub fn parse_mode<T, E>(
    mode: u8,
    rest: &[u8],
    header: &[u8],
    from_bytes: impl Fn(&[u8]) -> std::result::Result<T, E>,
    from_json: impl Fn(&str) -> std::result::Result<T, E>,
) -> Option<std::result::Result<T, E>> {
    match mode {
        0 => Some(from_bytes(rest)),
        1 => Some(from_bytes(&seal(header, rest))),
        _ => std::str::from_utf8(rest).ok().map(from_json),
    }
}

/// Round-trip a model through its binary and JSON formats, retaining target-specific equality.
pub fn round_trip<M>(
    model: &M,
    to_bytes: impl Fn(&M) -> Result<Vec<u8>>,
    from_bytes: impl Fn(&[u8]) -> Result<M>,
    to_json: impl Fn(&M) -> Result<String>,
    from_json: impl Fn(&str) -> Result<M>,
    eq: impl Fn(&M, &M),
) {
    let bytes = to_bytes(model).expect("an accepted model saves");
    let from_bytes = from_bytes(&bytes).expect("a saved model loads");
    let json = to_json(model).expect("an accepted model saves");
    let from_json = from_json(&json).expect("a saved model loads");
    eq(model, &from_bytes);
    eq(model, &from_json);
}

/// Shared one-parser targets for text and binary model parsers.
#[macro_export]
macro_rules! text_target {
    ($parser:path) => {
        libfuzzer_sys::fuzz_target!(|data: &[u8]| {
            let Ok(text) = std::str::from_utf8(data) else {
                return;
            };
            if let Ok(model) = $parser(text) {
                common::exercise(&model);
            }
        });
    };
}

#[macro_export]
macro_rules! bytes_target {
    ($parser:path) => {
        libfuzzer_sys::fuzz_target!(|data: &[u8]| {
            if let Ok(model) = $parser(data) {
                common::exercise(&model);
            }
        });
    };
}

const MAX_PREDICT_FEATURES: usize = 64;
const MAX_OUTPUTS: usize = 16;

/// Check compact predictions using the shared probe size ceilings.
pub fn exercise_compact(model: &CompactModel) {
    let n_features = model.n_features();
    let k = model.n_outputs();
    assert!(n_features > 0 && k > 0);
    for feature in model.used_features() {
        assert!(feature < n_features);
    }
    if !small_enough(n_features, k) {
        return;
    }
    let probe = probe_matrix(n_features);
    let margin = model
        .predict_margin(&probe)
        .expect("probe matrix matches the model");
    assert_eq!((margin.n_rows(), margin.width()), (probe.n_rows(), k));
    let preds = model
        .predict(&probe)
        .expect("probe matrix matches the model");
    let expected = if model.objective().name() == "multi:softmax" {
        1
    } else {
        k
    };
    assert_eq!((preds.n_rows(), preds.width()), (probe.n_rows(), expected));
}
/// Widest model `exercise` predicts with: a fuzzed header can claim any
/// feature or output count, which only makes the probe matrix (not the
/// model) expensive.
const MAX_INTERACTION_FEATURES: usize = 8;
/// Bitwise equality, so NaN predictions compare equal to themselves.
pub fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// A small probe matrix for `n_features` columns: all zeros, all missing,
/// and a row of mixed-sign, mixed-magnitude values.
pub fn probe_matrix(n_features: usize) -> DMatrix {
    let mut data = Vec::with_capacity(3 * n_features);
    data.extend(std::iter::repeat_n(0.0, n_features));
    data.extend(std::iter::repeat_n(f32::NAN, n_features));
    data.extend((0..n_features).map(|f| (f as f32 - 2.0) * 7.5_f32.powi(f as i32 % 5)));
    DMatrix::from_dense(&data, 3, n_features).expect("probe matrix is valid")
}

fn small_enough(n_features: usize, n_outputs: usize) -> bool {
    n_features <= MAX_PREDICT_FEATURES && n_outputs <= MAX_OUTPUTS
}

/// Runs every prediction and serialization path on `model`, which a parser
/// accepted or training produced. Accepting a model promises it is sound,
/// so none of these may panic, successful results must have the documented
/// shapes, and every lossless round trip must reproduce the model.
pub fn exercise(model: &BoostedModel) {
    let n_features = model.n_features();
    let k = model.n_outputs();
    assert!(n_features > 0);
    assert!(k > 0);

    for kind in [
        ImportanceType::Weight,
        ImportanceType::Gain,
        ImportanceType::TotalGain,
        ImportanceType::Cover,
        ImportanceType::TotalCover,
    ] {
        for &feature in model.feature_importance(kind).keys() {
            assert!(feature < n_features);
        }
    }

    let margin = small_enough(n_features, k).then(|| predict_all(model));

    // Native binary: decoding what was encoded re-encodes to the same bytes.
    let bytes = model.to_bytes().expect("a valid model encodes");
    let decoded = BoostedModel::from_bytes(&bytes).expect("an encoded model decodes");
    assert!(
        decoded.to_bytes().expect("re-encode") == bytes,
        "native round trip changed the model"
    );

    // Native JSON: the same model back.
    let json = model.to_json().expect("a valid model serializes to JSON");
    let from_json = BoostedModel::from_json(&json).expect("serialized JSON parses");
    assert!(
        from_json.to_bytes().expect("re-encode") == bytes,
        "JSON round trip changed the model"
    );

    // XGBoost interchange is partial; whatever exports must import and
    // predict like the model (see `check_xgboost_round_trip`).
    if let Ok(xgb) = model.to_xgboost_json() {
        let imported =
            BoostedModel::from_xgboost_json(&xgb).expect("exported XGBoost JSON imports");
        check_xgboost_round_trip(model, &imported, &xgb, "JSON");
        // Both encodings hold the same document.
        if let Ok(ubj) = model.to_xgboost_ubjson() {
            let imported =
                BoostedModel::from_xgboost_ubjson(&ubj).expect("exported XGBoost UBJSON imports");
            check_xgboost_round_trip(model, &imported, &xgb, "UBJSON");
        }
    }

    if let Ok(compact) = model.to_compact() {
        assert_eq!(compact.n_features(), n_features);
        assert_eq!(compact.n_outputs(), k);
        let reparsed = CompactModel::from_bytes(&compact.to_bytes()).expect("compact bytes parse");
        assert!(reparsed.to_bytes() == compact.to_bytes());
        if let Some(margin) = &margin {
            let data = probe_matrix(n_features);
            let compact_margin = compact
                .predict_margin(&data)
                .expect("compact model predicts");
            assert!(
                same_bits(margin.as_slice(), compact_margin.as_slice()),
                "compact margins differ"
            );
            let preds = model
                .predict(&data, Iterations::Best)
                .expect("model predicts");
            let compact_preds = compact.predict(&data).expect("compact model predicts");
            assert!(
                same_bits(preds.as_slice(), compact_preds.as_slice()),
                "compact predictions differ"
            );
        }
    }
}

/// Runs each prediction API on the probe matrix and returns the margins.
fn predict_all(model: &BoostedModel) -> Predictions {
    let n_features = model.n_features();
    let k = model.n_outputs();
    let data = probe_matrix(n_features);
    let n = data.n_rows();

    let margin = model
        .predict_margin(&data, Iterations::Best)
        .expect("probe matrix matches the model");
    assert_eq!((margin.n_rows(), margin.width()), (n, k));
    let whole = model
        .predict_margin(&data, ..)
        .expect("the whole model is a valid range");
    assert_eq!((whole.n_rows(), whole.width()), (n, k));

    let preds = model
        .predict(&data, Iterations::Best)
        .expect("probe matrix matches the model");
    let width = if model.objective().name() == "multi:softmax" {
        1
    } else {
        k
    };
    assert_eq!((preds.n_rows(), preds.width()), (n, width));
    model
        .predict_class(&data, Iterations::Best)
        .expect("probe matrix matches the model");

    let leaves = model
        .predict_leaf(&data, ..)
        .expect("probe matrix matches the model");
    assert_eq!((leaves.n_rows(), leaves.width()), (n, model.num_trees()));
    for row in leaves.rows() {
        for (tree, &leaf) in model.trees().iter().zip(row) {
            assert!(tree.nodes()[leaf as usize].is_leaf());
        }
    }

    if let Ok(contribs) = model.predict_contribs(&data, Iterations::Best) {
        assert_eq!(
            (
                contribs.n_rows(),
                contribs.n_outputs(),
                contribs.n_features()
            ),
            (n, k, n_features)
        );
    }
    if n_features <= MAX_INTERACTION_FEATURES
        && let Ok(interactions) = model.predict_interactions(&data, Iterations::Best)
    {
        assert_eq!(
            (
                interactions.n_rows(),
                interactions.n_outputs(),
                interactions.n_features()
            ),
            (n, k, n_features)
        );
    }
    if DistFamily::from_objective(model.objective().name()).is_some() {
        model
            .predict_distribution(&data, Iterations::Best)
            .expect("dist:* models predict distributions");
    }

    // Slicing every iteration keeps the whole model's predictions.
    if model.num_boost_rounds() > 0
        && let Ok(sliced) = model.slice(.., 1)
    {
        let sliced_margin = sliced
            .predict_margin(&data, Iterations::Best)
            .expect("a slice predicts");
        assert!(
            same_bits(whole.as_slice(), sliced_margin.as_slice()),
            "slice(.., 1) changed the margins"
        );
        let first = model.slice(..1, 1).expect("the first iteration slices");
        assert_eq!(first.num_boost_rounds(), 1);
    }
    margin
}

/// Checks `imported`, read back from the XGBoost export of `model` whose
/// JSON text is `exported`.
///
/// Trees, tree weights and the iteration layout travel losslessly: with the
/// intercepts replaced by a per-row `base_margin`, the margins are bitwise
/// equal. The intercepts travel in the space XGBoost stores `base_score` in
/// (through the objective's inverse link, and its link and clamps on the
/// way back), so they are compared there: exporting `imported` again must
/// store the same `base_score` up to `f32` rounding of the link and its
/// inverse (relative `1e-5`) and XGBoost's clamp of probabilities to
/// `[1e-6, 1 - 1e-6]` (absolute `2e-6`). A stored value the link maps past
/// `f32` (non-finite on the second export) is the format's limit and is not
/// compared.
fn check_xgboost_round_trip(
    model: &BoostedModel,
    imported: &BoostedModel,
    exported: &str,
    format: &str,
) {
    let n_features = model.n_features();
    let k = model.n_outputs();
    assert_eq!(imported.n_features(), n_features);
    assert_eq!(imported.n_outputs(), k);
    if small_enough(n_features, k) {
        let probe = probe_matrix(n_features);
        let intercepts = vec![0.0; probe.n_rows() * k];
        let data = probe
            .with_base_margin(&intercepts)
            .expect("zero base margins are valid");
        let before = model
            .predict_margin(&data, Iterations::Best)
            .expect("model predicts");
        let after = imported
            .predict_margin(&data, Iterations::Best)
            .expect("imported model predicts");
        assert!(
            same_bits(before.as_slice(), after.as_slice()),
            "XGBoost {format} round trip changed the trees: {before:?} -> {after:?}"
        );
    }
    let again = imported
        .to_xgboost_json()
        .expect("an imported XGBoost model exports again");
    let (stored, restored) = (stored_base_score(exported), stored_base_score(&again));
    assert_eq!(stored.len(), restored.len());
    for (&s0, &s1) in stored.iter().zip(&restored) {
        assert!(
            !s1.is_finite() || (s1 - s0).abs() <= 1e-5 * s0.abs() + 2e-6,
            "XGBoost {format} round trip changed the intercepts: stored {stored:?}, \
             re-exported {restored:?}"
        );
    }
}

/// The `base_score` entries an XGBoost JSON export stores
/// (`"base_score": "[v0,v1,...]"`, written once, in `learner_model_param`).
fn stored_base_score(json: &str) -> Vec<f32> {
    let key = r#""base_score": "["#;
    let start = json.find(key).expect("the export stores base_score") + key.len();
    let len = json[start..].find(']').expect("base_score is a vector");
    json[start..start + len]
        .split(',')
        .map(|v| v.parse().expect("base_score entries are numbers"))
        .collect()
}
