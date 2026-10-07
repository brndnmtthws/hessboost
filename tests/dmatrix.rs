//! `DMatrix` construction and clones: an owned dense `Vec` builds the same
//! matrix as the copied slice, the dense scan refuses infinities at any
//! position, and a clone stays independent of its source.

use hessboost::data::FeatureType;
use hessboost::data::target_stats::OrderedTargetEncoder;
use hessboost::model::Iterations;
use hessboost::prelude::*;

mod common;
use common::bits::same_bits;
use common::{four_features, invalid_data, with_threads};

/// Every cell's bit pattern, `None` where missing.
fn cells(d: &DMatrix) -> Vec<Option<u32>> {
    (0..d.n_rows())
        .flat_map(|r| (0..d.n_cols()).map(move |c| d.get(r, c).map(f32::to_bits)))
        .collect()
}

/// `n` rows of [`four_features`] (feature 3 partly missing) and labels.
fn table(n: usize) -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..n).flat_map(four_features).collect();
    let y = x
        .as_chunks::<4>()
        .0
        .iter()
        .map(|r| 2.0 * r[0] - r[1] + r[3].max(0.25))
        .collect();
    (x, y)
}

fn assert_refused(result: Result<DMatrix>) {
    assert_eq!(invalid_data(result), ("data", None));
}

#[test]
fn from_dense_vec_builds_the_from_dense_matrix() {
    let n = 400;
    let (x, y) = table(n);
    let copied = DMatrix::from_dense(&x, n, 4)
        .unwrap()
        .with_labels(&y)
        .unwrap();
    let owned = DMatrix::from_dense_vec(x.clone(), n, 4)
        .unwrap()
        .with_labels(&y)
        .unwrap();
    assert!(owned.missing().is_nan());
    assert_eq!(cells(&owned), cells(&copied));

    let params = TrainingParams::builder().max_depth(3).build().unwrap();
    let a = train(&params, &copied, 20).unwrap();
    let b = train(&params, &owned, 20).unwrap();
    for data in [&copied, &owned] {
        let pa = a.predict(data, Iterations::Best).unwrap();
        let pb = b.predict(data, Iterations::Best).unwrap();
        assert!(same_bits(pa.as_slice(), pb.as_slice()));
    }

    // The same refusals as `from_dense`.
    assert!(matches!(
        DMatrix::from_dense_vec(x.clone(), n + 1, 4),
        Err(HessboostError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        DMatrix::from_dense_vec(Vec::new(), 0, 4),
        Err(HessboostError::EmptyDataset(_))
    ));
    let mut bad = x;
    bad[7] = f32::INFINITY;
    assert_refused(DMatrix::from_dense_vec(bad, n, 4));
}

/// Every position of a matrix spanning several scan blocks, through each
/// dense constructor; NaN is missing under the default sentinel.
#[test]
fn dense_scan_refuses_infinities_at_any_position() {
    let (rows, cols) = (155, 5);
    let n = rows * cols;
    let base: Vec<f32> = (0..n).map(|i| (i % 13) as f32).collect();
    for pos in 0..n {
        for inf in [f32::INFINITY, f32::NEG_INFINITY] {
            let mut x = base.clone();
            x[pos] = inf;
            assert_refused(DMatrix::from_dense(&x, rows, cols));
            assert_refused(DMatrix::from_dense_with_missing(&x, rows, cols, -1.0));
            assert_refused(DMatrix::from_dense_vec(x, rows, cols));
        }
        let mut x = base.clone();
        x[pos] = f32::NAN;
        let d = DMatrix::from_dense(&x, rows, cols).unwrap();
        assert_eq!(d.get(pos / cols, pos % cols), None);
        assert_eq!(
            cells(&DMatrix::from_dense_vec(x, rows, cols).unwrap()),
            cells(&d)
        );
    }
    let all_missing = DMatrix::from_dense(&vec![f32::NAN; n], rows, cols).unwrap();
    assert!(cells(&all_missing).iter().all(Option::is_none));
}

/// Inputs large enough to be scanned in parallel blocks.
#[test]
fn parallel_dense_scan_refuses_infinities() {
    let cols = 8;
    let rows = (1 << 19) + 125;
    let n = rows * cols;
    let value = |i: usize| (i % 1000) as f32;
    let mut x: Vec<f32> = (0..n).map(value).collect();
    let positions = [0, 255, 256, (1 << 18) - 1, 1 << 18, n / 2, n - 1];
    with_threads(4, || {
        for pos in positions {
            for inf in [f32::INFINITY, f32::NEG_INFINITY] {
                x[pos] = inf;
                assert_refused(DMatrix::from_dense(&x, rows, cols));
                assert_refused(DMatrix::from_dense_with_missing(&x, rows, cols, -1.0));
            }
            x[pos] = value(pos);
        }
        for pos in positions {
            x[pos] = f32::NAN;
        }
        let d = DMatrix::from_dense(&x, rows, cols).unwrap();
        assert_eq!(d.get(0, 0), None);
        assert_eq!(d.get(rows - 1, cols - 2), Some(value(n - 2)));
        assert_eq!(d.get(rows - 1, cols - 1), None);
    });
}

/// A dense and a sparse matrix of the same table: a categorical column 0
/// (missing on every 9th row) and a numeric column 1, with labels.
fn categorical_pair() -> [DMatrix; 2] {
    let n = 300;
    let category = |i: usize| (!i.is_multiple_of(9)).then_some((i % 12) as f32);
    let numeric = |i: usize| (i % 17) as f32 / 17.0;
    let y: Vec<f32> = (0..n)
        .map(|i| category(i).unwrap_or(3.0) + numeric(i))
        .collect();
    let x: Vec<f32> = (0..n)
        .flat_map(|i| [category(i).unwrap_or(f32::NAN), numeric(i)])
        .collect();
    let (mut indptr, mut indices, mut values) = (vec![0], Vec::new(), Vec::new());
    for i in 0..n {
        if let Some(c) = category(i) {
            indices.push(0);
            values.push(c);
        }
        indices.push(1);
        values.push(numeric(i));
        indptr.push(values.len());
    }
    let types = [FeatureType::Categorical, FeatureType::Numerical];
    [
        DMatrix::from_dense(&x, n, 2).unwrap(),
        DMatrix::from_csr(indptr, indices, values, 2).unwrap(),
    ]
    .map(|d| {
        d.with_labels(&y)
            .unwrap()
            .with_feature_types(&types)
            .unwrap()
    })
}

#[test]
fn target_stats_transform_leaves_a_clone_and_its_source_unchanged() {
    let encoder = OrderedTargetEncoder::builder().seed(3).build().unwrap();
    for source in categorical_pair() {
        let before = cells(&source);
        let (encoded_train, fitted) = encoder.fit_transform(&source, &[0]).unwrap();
        let clone = source.clone();
        let encoded = fitted.transform(&clone).unwrap();
        assert_ne!(cells(&encoded), before);
        assert_ne!(cells(&encoded_train), before);
        assert_eq!(cells(&source), before);
        assert_eq!(cells(&clone), before);
        assert_eq!(clone.feature_types()[0], FeatureType::Categorical);
        // Encoding the source again gives the clone's encoding.
        assert_eq!(cells(&fitted.transform(&source).unwrap()), cells(&encoded));
    }
}

#[test]
fn metadata_setters_leave_a_clone_and_its_source_unchanged() {
    for source in categorical_pair() {
        let n = source.n_rows();
        let labels = source.labels().unwrap().to_vec();
        let values = cells(&source);
        let changed = source
            .clone()
            .with_labels(&vec![1.0; n])
            .unwrap()
            .with_weights(&vec![2.0; n])
            .unwrap()
            .with_base_margin(&vec![0.5; n])
            .unwrap()
            .with_group_sizes(&[n])
            .unwrap()
            .with_label_bounds(&vec![0.0; n], &vec![1.0; n])
            .unwrap()
            .with_feature_weights(&[1.0, 3.0])
            .unwrap()
            .with_feature_types(&[FeatureType::Numerical; 2])
            .unwrap();
        assert_eq!(changed.labels().unwrap(), vec![1.0; n]);
        assert_eq!(cells(&changed), values);
        assert_eq!(source.labels().unwrap(), labels);
        assert!(source.weights().is_none());
        assert!(source.base_margin().is_none());
        assert!(source.group().is_none());
        assert!(source.label_lower_bound().is_none());
        assert!(source.feature_weights().is_none());
        assert_eq!(source.feature_types()[0], FeatureType::Categorical);

        // Changing the source leaves an earlier clone as it was.
        let clone = source.clone();
        let source = source.with_labels(&vec![-1.0; n]).unwrap();
        assert_eq!(clone.labels().unwrap(), labels);
        assert_eq!(source.labels().unwrap(), vec![-1.0; n]);
        assert_eq!(cells(&clone), values);
    }
}
