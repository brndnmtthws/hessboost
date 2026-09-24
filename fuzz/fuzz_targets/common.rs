//! Model checks shared by the fuzz targets. Each target is its own `[[bin]]`,
//! so this file is pulled in per target with `#[path]`.
#![allow(dead_code, reason = "each target uses a different subset")]

use hessboost::model::ImportanceType;
use hessboost::model::compact::CompactModel;
use hessboost::objective::distributional::DistFamily;
use hessboost::prelude::*;

/// Widest model `exercise` predicts with: a fuzzed header can claim any
/// feature or output count, which only makes the probe matrix (not the
/// model) expensive.
const MAX_PREDICT_FEATURES: usize = 64;
const MAX_OUTPUTS: usize = 16;
/// Interaction values are `(n_features + 1)^2` per row and output.
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
            assert!((feature as usize) < n_features);
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

    // XGBoost interchange is partial; whatever exports must import.
    if let Ok(xgb) = model.to_xgboost_json() {
        let imported =
            BoostedModel::from_xgboost_json(&xgb).expect("exported XGBoost JSON imports");
        assert_eq!(imported.n_features(), n_features);
        assert_eq!(imported.n_outputs(), k);
    }
    if let Ok(ubj) = model.to_xgboost_ubjson() {
        let imported =
            BoostedModel::from_xgboost_ubjson(&ubj).expect("exported XGBoost UBJSON imports");
        assert_eq!(imported.n_features(), n_features);
        assert_eq!(imported.n_outputs(), k);
    }

    // The compact layout predicts bit-identically to its source.
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
            assert!(same_bits(margin, &compact_margin), "compact margins differ");
            let preds = model.predict(&data).expect("model predicts");
            let compact_preds = compact.predict(&data).expect("compact model predicts");
            assert!(
                same_bits(&preds, &compact_preds),
                "compact predictions differ"
            );
        }
    }
}

/// Runs each prediction API on the probe matrix and returns the margins.
fn predict_all(model: &BoostedModel) -> Vec<f32> {
    let n_features = model.n_features();
    let k = model.n_outputs();
    let data = probe_matrix(n_features);
    let n = data.n_rows();

    let margin = model
        .predict_margin(&data)
        .expect("probe matrix matches the model");
    assert_eq!(margin.len(), n * k);
    let whole = model
        .predict_margin_range(&data, ..)
        .expect("the whole model is a valid range");
    assert_eq!(whole.len(), n * k);

    let preds = model
        .predict(&data)
        .expect("probe matrix matches the model");
    let expected = if model.objective() == "multi:softmax" {
        n
    } else {
        n * k
    };
    assert_eq!(preds.len(), expected);
    model
        .predict_class(&data)
        .expect("probe matrix matches the model");

    let leaves = model
        .predict_leaf(&data)
        .expect("probe matrix matches the model");
    assert_eq!(leaves.len(), n * model.num_trees());
    for (i, &leaf) in leaves.iter().enumerate() {
        let tree = &model.trees()[i % model.num_trees()];
        assert!(tree.nodes()[leaf as usize].is_leaf());
    }

    if let Ok(contribs) = model.predict_contribs(&data) {
        assert_eq!(contribs.len(), n * k * (n_features + 1));
    }
    if n_features <= MAX_INTERACTION_FEATURES
        && let Ok(interactions) = model.predict_interactions(&data)
    {
        assert_eq!(
            interactions.len(),
            n * k * (n_features + 1) * (n_features + 1)
        );
    }
    if DistFamily::from_objective(model.objective()).is_some() {
        model
            .predict_distribution(&data)
            .expect("dist:* models predict distributions");
    }

    // Slicing every iteration keeps the whole model's predictions.
    if model.num_boost_rounds() > 0
        && let Ok(sliced) = model.slice(.., 1)
    {
        let sliced_margin = sliced.predict_margin(&data).expect("a slice predicts");
        assert!(
            same_bits(&whole, &sliced_margin),
            "slice(.., 1) changed the margins"
        );
        let first = model.slice(..1, 1).expect("the first iteration slices");
        assert_eq!(first.num_boost_rounds(), 1);
    }
    margin
}
