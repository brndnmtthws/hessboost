//! `ModelFormat::detect` and `DiffusionFormat::detect`: every format a
//! model writes (and the XGBoost, LightGBM and saved-release files) is
//! recognized as itself, and bytes that are none of them are recognized as
//! nothing rather than guessed as the native binary format.

use std::num::NonZeroUsize;
use std::path::Path;

use hessboost::diffusion::{DiffusionFormat, DiffusionModel, DiffusionParams};
use hessboost::prelude::*;

/// Byte strings no reader accepts; none may be taken for a format.
const GARBAGE: &[&[u8]] = &[
    b"",
    b"   \n",
    b"garbage",
    b"\x00\x01\x02\x03",
    b"[1, 2, 3]",
    b"{x",
    b"{ = }",
    b"tree",
    b"trees are green\n",
    b"HBM",
];

fn data(name: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(name),
    )
    .unwrap()
}

#[test]
fn model_formats_are_detected_from_their_bytes() {
    let x: Vec<f32> = (0..60).map(|i| (i % 13) as f32).collect();
    let dtrain = DMatrix::from_dense(&x, 30, 2)
        .unwrap()
        .with_labels(&x[..30])
        .unwrap();
    let model = train(&TrainingParams::default(), &dtrain, 3).unwrap();
    for format in [
        ModelFormat::Binary,
        ModelFormat::Json,
        ModelFormat::XgboostJson,
        ModelFormat::XgboostUbjson,
    ] {
        let bytes = model.encode(format).unwrap();
        assert_eq!(ModelFormat::detect(&bytes), Some(format), "{format:?}");
    }
    let files = [
        ("xgboost-3.4.2-categorical.json", ModelFormat::XgboostJson),
        ("xgboost-3.4.2-categorical.ubj", ModelFormat::XgboostUbjson),
        ("lightgbm-4.7.0-binary.txt", ModelFormat::LightgbmText),
        ("saved/0.2.0/dart.bin", ModelFormat::Binary),
        ("saved/0.2.0/dart.json", ModelFormat::Json),
    ];
    for (name, format) in files {
        let bytes = data(name);
        assert_eq!(ModelFormat::detect(&bytes), Some(format), "{name}");
        assert!(BoostedModel::decode(&bytes, format).is_ok(), "{name}");
    }
    // UBJSON allows `N` no-ops before a key, the first one included.
    let mut padded = data("xgboost-3.4.2-categorical.ubj");
    assert_eq!(padded[0], b'{');
    padded.insert(1, b'N');
    assert_eq!(
        ModelFormat::detect(&padded),
        Some(ModelFormat::XgboostUbjson)
    );
    assert!(BoostedModel::decode(&padded, ModelFormat::XgboostUbjson).is_ok());
    for garbage in GARBAGE {
        assert_eq!(ModelFormat::detect(garbage), None, "{garbage:?}");
    }
    // Import only: LightGBM text decodes but does not encode.
    assert!(matches!(
        model.encode(ModelFormat::LightgbmText),
        Err(HessboostError::InvalidParameter { name: "format", .. })
    ));
}

#[test]
fn diffusion_formats_are_detected_from_their_bytes() {
    let x: Vec<f32> = (0..40).map(|i| i as f32 / 40.0).collect();
    let y: Vec<f32> = x.iter().map(|v| 2.0 * v).collect();
    let data = DMatrix::from_dense(&x, 40, 1)
        .unwrap()
        .with_labels(&y)
        .unwrap();
    let mut params = DiffusionParams::default();
    params.n_repeats = NonZeroUsize::new(2).unwrap();
    params.num_boost_round = NonZeroUsize::new(5).unwrap();
    params.residualizer = None;
    let model = DiffusionModel::fit(&params, &data).unwrap();
    for format in [DiffusionFormat::Binary, DiffusionFormat::Json] {
        let bytes = model.encode(format).unwrap();
        assert_eq!(DiffusionFormat::detect(&bytes), Some(format), "{format:?}");
        let decoded = DiffusionModel::decode(&bytes, format).unwrap();
        assert_eq!(decoded.encode(format).unwrap(), bytes, "{format:?}");
    }
    for garbage in GARBAGE {
        assert_eq!(DiffusionFormat::detect(garbage), None, "{garbage:?}");
    }
}
