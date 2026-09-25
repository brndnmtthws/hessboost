//! Property-based tests (proptest) over randomized data and configs.
//!
//! These lock in invariants that unit tests only spot-check: training
//! determinism (also across thread counts), dense/sparse prediction
//! equivalence, monotone-constraint guarantees, and lossless model
//! round-trips.

use hessboost::config::Monotone;
use hessboost::prelude::*;
use proptest::prelude::*;

mod common;
use common::labeled_dense;

const ROWS: usize = 40;
const COLS: usize = 3;

/// Strategy for a `ROWS × COLS` dense feature matrix and a label vector, with
/// bounded, finite values.
fn dataset() -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    (
        prop::collection::vec(-10.0f32..10.0, ROWS * COLS),
        prop::collection::vec(-5.0f32..5.0, ROWS),
    )
}

fn base_params(seed: u64) -> TrainingParams {
    TrainingParams::builder()
        .objective("reg:squarederror")
        .max_depth(3)
        .eta(0.3)
        .subsample(0.8) // exercise the RNG path
        .colsample_bytree(0.8)
        .seed(seed)
        .build()
        .unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Same data, params, and seed must yield identical predictions.
    #[test]
    fn training_is_deterministic((x, y) in dataset(), seed in 0u64..10_000) {
        let d = labeled_dense(&x, COLS, &y);
        let params = base_params(seed);
        let a = train(&params, &d, 15).unwrap().predict(&d).unwrap();
        let b = train(&params, &d, 15).unwrap().predict(&d).unwrap();
        prop_assert_eq!(a, b);
    }

    /// A dense matrix and the CSR matrix listing all the same entries must
    /// predict identically (both have every feature present).
    #[test]
    fn dense_equals_sparse((x, y) in dataset(), seed in 0u64..10_000) {
        let dense = labeled_dense(&x, COLS, &y);

        // Build an equivalent CSR that explicitly lists every entry.
        let mut indptr = vec![0usize];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for r in 0..ROWS {
            for c in 0..COLS {
                indices.push(c as u32);
                values.push(x[r * COLS + c]);
            }
            indptr.push(values.len());
        }
        let sparse = DMatrix::from_csr(indptr, indices, values, COLS)
            .unwrap()
            .with_labels(&y)
            .unwrap();

        let params = base_params(seed);
        let pd = train(&params, &dense, 15).unwrap().predict(&dense).unwrap();
        let ps = train(&params, &sparse, 15).unwrap().predict(&sparse).unwrap();
        for (a, b) in pd.iter().zip(&ps) {
            prop_assert!((a - b).abs() < 1e-4, "dense {a} vs sparse {b}");
        }
    }

    /// An increasing monotone constraint must produce predictions that never
    /// decrease as the constrained feature increases.
    #[test]
    fn monotone_increasing_holds(
        x in prop::collection::vec(-10.0f32..10.0, ROWS),
        y in prop::collection::vec(-5.0f32..5.0, ROWS),
    ) {
        let d = labeled_dense(&x, 1, &y);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .max_depth(4)
            .eta(0.3)
            .monotone_constraints(vec![Monotone::Increasing])
            .build()
            .unwrap();
        let model = train(&params, &d, 20).unwrap();
        let preds = model.predict(&d).unwrap();

        // Sort rows by feature value; predictions must be non-decreasing.
        let mut idx: Vec<usize> = (0..ROWS).collect();
        idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
        let mut prev = f32::NEG_INFINITY;
        for &i in &idx {
            prop_assert!(preds[i] >= prev - 1e-4, "monotonicity broken: {} < {}", preds[i], prev);
            prev = prev.max(preds[i]);
        }
    }

    /// Binary and native model serialization must be lossless w.r.t. predictions.
    #[test]
    fn serde_roundtrip_preserves_predictions((x, y) in dataset(), seed in 0u64..10_000) {
        let d = labeled_dense(&x, COLS, &y);
        let model = train(&base_params(seed), &d, 12).unwrap();
        let before = model.predict(&d).unwrap();

        let restored = BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
        let after = restored.predict(&d).unwrap();
        prop_assert_eq!(&before, &after);

        let from_json = BoostedModel::from_json(&model.to_json().unwrap()).unwrap();
        let after_json = from_json.predict(&d).unwrap();
        for (a, b) in before.iter().zip(&after_json) {
            prop_assert!((a - b).abs() < 1e-6);
        }
    }
}

/// Training is independent of the thread count where `f64` histogram sums
/// round differently under another grouping. With `base_score = 0`, squared
/// error's gradients are the negated labels: `2^54` at row 0, `-2^54` at
/// row 4096, `1` at row 4097, `0` elsewhere. In row order they sum to `1`;
/// split at row 4096 (as two worker threads once split these 8,192 sparse
/// rows), `-2^54 + 1` rounds to `-2^54` and the sum is `0`. The rows of bin
/// 0 carry those gradients, so its sum sets the split's leaf values.
#[test]
fn training_is_independent_of_the_thread_count() {
    let n = 8192;
    // Only feature 0 of 3 is present: a sparse index, with no column copy.
    let indptr: Vec<usize> = (0..=n).collect();
    let values: Vec<f32> = (0..n).map(|r| f32::from(r >= 8000)).collect();
    let mut labels = vec![0.0f32; n];
    labels[0] = -(2f32.powi(54));
    labels[4096] = 2f32.powi(54);
    labels[4097] = -1.0;
    let data = DMatrix::from_csr(indptr, vec![0; n], values, 3)
        .unwrap()
        .with_labels(&labels)
        .unwrap();
    let params = TrainingParams::builder()
        .objective("reg:squarederror")
        .base_score(0.0)
        .max_depth(2)
        .build()
        .unwrap();
    let train_on = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| train(&params, &data, 2).unwrap().to_bytes().unwrap())
    };
    let serial = train_on(1);
    for threads in [2, 4, 8] {
        assert!(train_on(threads) == serial, "{threads} threads");
    }
}
