use super::bitstream::{BitWriter, f16_exact, f16_to_f32};
use super::decode::{decode_threshold, split_frame};
use super::encode::{encode_threshold, frame, numeric_encoding};
use super::*;
use crate::config::{BoosterKind, Dart, GrowPolicy, LinearTree, TrainingParams, TreeMethod};
use crate::data::FeatureType;
use crate::objective::{Multiclass, Objective, Quantiles, RegLoss};
use crate::test_support::labeled_dense;
use crate::training::{Trainer, train};

/// Deterministic pseudo-random value in `[0, 1)`.
fn noise(i: usize) -> f32 {
    let mut z = (i as u64).wrapping_mul(crate::rng::GOLDEN);
    z ^= z >> 31;
    z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (z >> 40) as f32 / (1u64 << 24) as f32
}

/// `n × f` features with ~10% missing values and a nonlinear target.
fn dataset(n: usize, f: usize, missing: bool) -> (Vec<f32>, Vec<f32>) {
    let mut x: Vec<f32> = (0..n * f).map(|i| noise(i) * 10.0 - 3.0).collect();
    if missing {
        for (i, v) in x.iter_mut().enumerate() {
            if noise(i + 7_777_777) < 0.1 {
                *v = f32::NAN;
            }
        }
    }
    let y = x
        .chunks(f)
        .map(|r| {
            let a = if r[0].is_nan() { 1.0 } else { r[0] };
            let b = if r[1].is_nan() { -1.0 } else { r[1] };
            (a * b).sin() + 0.3 * a
        })
        .collect();
    (x, y)
}

/// [`dataset`] as a labeled dense matrix.
fn labeled(n: usize, f: usize, missing: bool) -> DMatrix {
    let (x, y) = dataset(n, f, missing);
    labeled_dense(&x, n, f, &y)
}

fn assert_bit_identical(model: &BoostedModel, data: &DMatrix) -> CompactModel {
    let compact = CompactModel::from_bytes(&model.to_compact_bytes().unwrap()).unwrap();
    let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    assert_eq!(
        bits(compact.predict_margin(data).unwrap().into_vec()),
        bits(model.predict_margin(data).unwrap().into_vec())
    );
    assert_eq!(
        bits(compact.predict(data).unwrap().into_vec()),
        bits(model.predict(data).unwrap().into_vec())
    );
    compact
}

#[test]
fn round_trip_is_bit_identical_across_model_kinds() {
    let (x, y) = dataset(600, 5, true);
    let data = labeled_dense(&x, 600, 5, &y);
    let labels01: Vec<f32> = y.iter().map(|&v| f32::from(v > 0.5)).collect();
    let labels3: Vec<f32> = y
        .iter()
        .map(|&v| ((v + 1.5).clamp(0.0, 2.9)) as u32 as f32)
        .collect();
    let binary = data.clone().with_labels(&labels01).unwrap();
    let multi = data.clone().with_labels(&labels3).unwrap();
    let matrix: Vec<f32> = y.iter().flat_map(|&v| [v, 1.0 - 2.0 * v]).collect();
    let targets2 = data.clone().with_label_matrix(&matrix, 2).unwrap();

    let base = || TrainingParams::builder().max_depth(4).eta(0.2);
    let classes = || Multiclass::new(3).unwrap();
    let cases: Vec<(TrainingParams, &DMatrix)> = vec![
        (base().build().unwrap(), &data),
        (
            base()
                .tree_method(TreeMethod::Exact)
                .objective(Objective::BinaryLogistic(RegLoss::default()))
                .build()
                .unwrap(),
            &binary,
        ),
        (
            base()
                .tree_method(TreeMethod::Approx)
                .objective(Objective::Softprob(classes()))
                .build()
                .unwrap(),
            &multi,
        ),
        (
            base()
                .objective(Objective::Softmax(classes()))
                .build()
                .unwrap(),
            &multi,
        ),
        (
            base()
                .booster(BoosterKind::Dart(
                    Dart::builder().rate_drop(0.3).build().unwrap(),
                ))
                .build()
                .unwrap(),
            &data,
        ),
        (
            base()
                .toad_penalty_feature(2.0)
                .toad_penalty_threshold(0.5)
                .build()
                .unwrap(),
            &data,
        ),
        // Iteration-major forests: tree `t` feeds output
        // `(t / num_parallel_tree) % n_outputs`.
        (
            base()
                .objective(Objective::Softprob(classes()))
                .num_parallel_tree(2)
                .build()
                .unwrap(),
            &multi,
        ),
        (base().num_parallel_tree(2).build().unwrap(), &targets2),
        (
            base()
                .objective(Objective::Quantile(Quantiles::new([0.2, 0.8]).unwrap()))
                .build()
                .unwrap(),
            &data,
        ),
    ];
    for (params, d) in cases {
        let model = train(&params, d, 15).unwrap();
        assert_bit_identical(&model, d);
    }
}

#[test]
fn deep_unbalanced_trees_use_the_preorder_layout() {
    let data = labeled(800, 4, true);
    let params = TrainingParams::builder()
        .grow_policy(GrowPolicy::LossGuide)
        .unlimited_depth()
        .max_leaves(48)
        .build()
        .unwrap();
    let model = train(&params, &data, 5).unwrap();
    let compact = assert_bit_identical(&model, &data);
    assert!(
        compact
            .trees
            .iter()
            .any(|t| matches!(t.layout, Layout::Preorder { .. })),
        "the case must exercise the preorder layout"
    );
}

#[test]
fn categorical_csr_and_base_margin_inputs_match() {
    let n = 500;
    let mut x = Vec::with_capacity(n * 3);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let cat = (noise(i) * 9.0) as u32 as f32;
        let v = noise(i + 99_999) * 4.0;
        x.extend_from_slice(&[cat, v, noise(i + 5) * 2.0]);
        y.push(
            if [1.0, 4.0, 6.0].contains(&cat) {
                2.0
            } else {
                0.0
            } + v,
        );
    }
    let data = DMatrix::from_dense(&x, n, 3)
        .unwrap()
        .with_feature_types(&[
            FeatureType::Categorical,
            FeatureType::Numerical,
            FeatureType::Numerical,
        ])
        .unwrap()
        .with_labels(&y)
        .unwrap();
    let params = TrainingParams::builder().max_depth(3).build().unwrap();
    let model = train(&params, &data, 10).unwrap();
    assert!(
        model
            .trees()
            .iter()
            .any(|t| t.nodes().iter().any(|n| n.is_categorical))
    );
    assert_bit_identical(&model, &data);

    // The same rows as CSR (dropping feature 2 when it is small) and with
    // per-row base margins.
    let (mut indptr, mut indices, mut values) = (vec![0], Vec::new(), Vec::new());
    for r in x.chunks(3) {
        for (f, &v) in r.iter().enumerate() {
            if f != 2 || v > 0.5 {
                indices.push(f as u32);
                values.push(v);
            }
        }
        indptr.push(indices.len());
    }
    let csr = DMatrix::from_csr(indptr, indices, values, 3)
        .unwrap()
        .with_base_margin(&y.iter().map(|v| v * 0.1).collect::<Vec<_>>())
        .unwrap();
    assert_bit_identical(&model, &csr);
}

#[test]
fn only_the_early_stopped_prefix_is_stored() {
    let data = labeled(400, 3, false);
    let (xv, yv) = dataset(200, 3, false);
    let yv: Vec<f32> = yv.iter().map(|v| -v).collect();
    let valid = labeled_dense(&xv, 200, 3, &yv);
    let params = TrainingParams::builder().build().unwrap();
    let model = Trainer::new(&params, &data, 50)
        .eval(&valid, "valid")
        .early_stopping_rounds(std::num::NonZeroUsize::new(3).unwrap())
        .train()
        .unwrap()
        .model;
    let best = model.best_iteration().expect("early stopping triggers");
    let compact = assert_bit_identical(&model, &data);
    assert_eq!(compact.num_trees(), best + 1);
}

#[test]
fn thresholds_take_the_narrowest_exact_encoding() {
    let cases: [(&[f32], ThresholdKind, u32); 7] = [
        (&[0.0, 1.0], ThresholdKind::Unsigned, 0),
        (&[3.0, 200.0], ThresholdKind::Unsigned, 3),
        (&[-2.0, 1.0], ThresholdKind::Signed, 1),
        (&[-3.0, 1.0], ThresholdKind::Signed, 2),
        (&[0.5, -0.0, 100.25], ThresholdKind::Float, 4),
        (&[0.1], ThresholdKind::Float, 5),
        (&[f32::MIN, 3.0], ThresholdKind::Float, 5),
    ];
    for (values, kind, code) in cases {
        assert_eq!(numeric_encoding(values), (kind, code), "{values:?}");
        let width = 1 << code;
        for &v in values {
            let raw = encode_threshold(v, kind, width) as u32;
            assert_eq!(
                decode_threshold(raw, kind, width).unwrap().to_bits(),
                v.to_bits()
            );
        }
    }
    // binary16 subnormals and the largest finite value are exact.
    for v in [
        f32::from_bits(0x3380_0000),
        3.0 * f32::from_bits(0x3380_0000),
        65504.0,
    ] {
        assert_eq!(f16_to_f32(f16_exact(v).unwrap()).to_bits(), v.to_bits());
    }
    assert!(f16_exact(65520.0).is_none());
    assert!(f16_exact(1.0 + f32::EPSILON).is_none());
}

#[test]
fn compact_is_far_smaller_than_native() {
    let data = labeled(2000, 8, false);
    let params = TrainingParams::builder().max_depth(4).build().unwrap();
    let model = train(&params, &data, 100).unwrap();
    let report = model.size_report().unwrap();
    assert_eq!(report.trees, 100);
    assert_eq!(
        report.compact_bytes,
        model.to_compact_bytes().unwrap().len()
    );
    // The native format is zstd-compressed; the bit-packed layout still
    // beats it clearly (about 2.2x here with libzstd's default level).
    assert!(
        report.compression_ratio() > 1.5,
        "compact {} vs native {}",
        report.compact_bytes,
        report.native_bytes
    );
    assert!(report.reuse_factor() >= 1.0);
}

#[test]
fn corrupt_input_is_rejected_without_panicking() {
    let data = labeled(300, 4, true);
    let params = TrainingParams::builder().max_depth(3).build().unwrap();
    let model = train(&params, &data, 5).unwrap();
    let bytes = model.to_compact_bytes().unwrap();
    for len in 0..bytes.len() {
        assert!(
            CompactModel::from_bytes(&bytes[..len]).is_err(),
            "prefix {len}"
        );
    }
    let mut longer = bytes.clone();
    longer.push(0);
    assert!(CompactModel::from_bytes(&longer).is_err());
    // Single bit flips either fail to parse or give a model that predicts.
    for i in 0..bytes.len() * 8 {
        let mut flipped = bytes.clone();
        flipped[i / 8] ^= 1 << (i % 8);
        if let Ok(m) = CompactModel::from_bytes(&flipped) {
            let _ = m.predict_margin(&data);
        }
    }
}

/// `bytes` with its metadata replaced by `edit` applied to the decoded
/// metadata.
fn with_meta(bytes: &[u8], edit: impl FnOnce(&mut Meta)) -> Vec<u8> {
    let (meta, stream) = split_frame(bytes).unwrap();
    let mut meta = Meta::decode(meta).unwrap();
    edit(&mut meta);
    frame(&meta, stream)
}

#[test]
fn inconsistent_layout_metadata_is_rejected() {
    let (x, y) = dataset(200, 3, false);
    let data = DMatrix::from_dense(&x, 200, 3)
        .unwrap()
        .with_label_matrix(&[y.clone(), y].concat(), 2)
        .unwrap();
    let params = TrainingParams::builder().max_depth(2).build().unwrap();
    let bytes = train(&params, &data, 0)
        .unwrap()
        .to_compact_bytes()
        .unwrap();
    assert!(CompactModel::from_bytes(&with_meta(&bytes, |_| ())).is_ok());
    // Two outputs × 2^63 parallel trees overflows the trees per iteration.
    let overflow = with_meta(&bytes, |m| m.num_parallel_tree = 1 << 63);
    assert!(matches!(
        CompactModel::from_bytes(&overflow),
        Err(HessboostError::ModelFormat(_))
    ));
    // A three-alpha objective cannot describe a two-output layout.
    let widened = with_meta(&bytes, |m| {
        m.objective = ModelObjective::trained_with(&Objective::Quantile(
            Quantiles::new([0.1, 0.5, 0.9]).unwrap(),
        ));
        m.n_targets = 1;
    });
    assert!(matches!(
        CompactModel::from_bytes(&widened),
        Err(HessboostError::ModelFormat(_))
    ));
}

/// `n_trees` complete depth-24 heaps whose split and leaf references are
/// zero bits wide: each tree costs only its seven header bits, so the
/// whole model is a few KB although it names 2^25 slots per tree.
fn zero_width_heaps(n_trees: usize, n_leaves: u64) -> Vec<u8> {
    let mut w = BitWriter::default();
    w.write(1, 32); // n_features
    w.write(1, 32); // n_outputs
    w.write_f32(0.5); // base score
    w.write(n_trees as u64, 32);
    w.write_bool(false); // no tree weights
    w.write(u64::from(DefaultDirection::AllLeft as u32), 2);
    w.write(1, 32); // used features
    w.write(1, 32); // max thresholds
    w.write(n_leaves, 32);
    w.write(5, 6); // heap depth bits
    w.write(0, 6); // preorder node-count bits
    w.write(u64::from(ThresholdKind::Unsigned as u32), 2);
    w.write(0, 3); // one-bit thresholds
    w.write(1, 1); // threshold 1
    for _ in 0..n_leaves {
        w.write_f32(1.0);
    }
    for _ in 0..n_trees {
        w.write_bool(false); // heap
        w.write(24, 5);
        w.write_bool(true); // complete
    }
    let meta = Meta {
        objective: ModelObjective::trained_with(&Objective::SquaredError(RegLoss::default())),
        max_delta_step: 0.0,
        num_class: 0,
        n_targets: 1,
        num_parallel_tree: 1,
        shrinkage: None,
    };
    frame(&meta, &w.bytes)
}

/// Validation work is bounded by the input: 4096 zero-width depth-24
/// heaps (under 4 KB) once cost 2^25 slot checks each.
#[test]
fn zero_width_heaps_validate_in_bounded_time() {
    let bytes = zero_width_heaps(4096, 1);
    assert!(bytes.len() < 4096);
    let model = CompactModel::from_bytes(&bytes).unwrap();
    let data = DMatrix::from_dense(&[0.0], 1, 1).unwrap();
    assert_eq!(model.predict_margin(&data).unwrap().as_slice(), [4096.5]);
    // The shared zero-width leaf reference is still range-checked.
    assert!(matches!(
        CompactModel::from_bytes(&zero_width_heaps(4096, 0)),
        Err(HessboostError::ModelFormat(_))
    ));
}

/// gblinear models and linear leaves have no compact encoding: refused,
/// never flattened to their constant fallback.
#[test]
fn gblinear_and_linear_leaf_models_are_rejected() {
    let data = labeled(200, 3, false);
    for params in [
        TrainingParams::builder().booster(BoosterKind::GbLinear),
        TrainingParams::builder()
            .max_depth(3)
            .linear_tree(LinearTree::default()),
    ] {
        let model = train(&params.build().unwrap(), &data, 3).unwrap();
        assert!(matches!(
            model.to_compact_bytes(),
            Err(HessboostError::ModelFormat(_))
        ));
    }
}
