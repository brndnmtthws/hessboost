//! Row predictions allocate nothing: once a model has laid out its trees
//! (its first prediction), predicting a row through every row method of
//! every model kind, and transforming its margins, allocates no memory.
//! Its own test binary, since it installs a counting global allocator.

use hessboost::prelude::*;
use serde_json::{Value, json};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

/// The system allocator, counting the calling thread's allocations.
struct Counting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: every call forwards to `System` unchanged; the counter is a
// const-initialized thread-local without a destructor, so touching it never
// allocates or fails.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
        // SAFETY: the caller's contract for `alloc` is `System.alloc`'s.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System.alloc` with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The allocations `f` makes on this thread.
fn allocations(f: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.with(Cell::get);
    f();
    ALLOCATIONS.with(Cell::get) - before
}

/// One model kind: its parameters, feature count, and class count (`0` for
/// a continuous label).
struct Kind {
    name: &'static str,
    params: Vec<(&'static str, Value)>,
    n_cols: usize,
    classes: usize,
}

fn model(kind: &Kind) -> (BoostedModel, Vec<f32>) {
    let (n_cols, classes) = (kind.n_cols, kind.classes);
    let n = 300;
    let x: Vec<f32> = (0..n * n_cols)
        .map(|i| {
            if i % 7 == 3 {
                f32::NAN
            } else {
                ((i * 37) % 101) as f32 / 25.0
            }
        })
        .collect();
    let y: Vec<f32> = x
        .chunks(n_cols)
        .map(|row| {
            let v = row.iter().filter(|v| !v.is_nan()).sum::<f32>();
            if classes > 0 {
                (v as usize % classes) as f32
            } else {
                v
            }
        })
        .collect();
    let dtrain = DMatrix::from_dense(&x, n, n_cols)
        .unwrap()
        .with_labels(&y)
        .unwrap();
    let params = TrainingParams::from_xgboost(kind.params.iter().cloned()).unwrap();
    (train(&params, &dtrain, 20).unwrap(), x)
}

#[test]
fn row_predictions_allocate_nothing() {
    let kind = |name, params, n_cols, classes| Kind {
        name,
        params,
        n_cols,
        classes,
    };
    let kinds = [
        kind("squared error", vec![], 8, 0),
        kind("wide rows", vec![("max_depth", json!(3))], 400, 0),
        kind(
            "softmax, 70 classes",
            vec![
                ("objective", json!("multi:softmax")),
                ("num_class", json!(70)),
                ("max_depth", json!(2)),
            ],
            8,
            70,
        ),
        kind(
            "softprob, vector leaves",
            vec![
                ("objective", json!("multi:softprob")),
                ("num_class", json!(3)),
                ("multi_strategy", json!("multi_output_tree")),
            ],
            8,
            3,
        ),
        kind(
            "quantiles",
            vec![
                ("objective", json!("reg:quantileerror")),
                ("quantile_alpha", json!([0.2, 0.8])),
            ],
            8,
            0,
        ),
        kind("linear leaves", vec![("linear_tree", json!(true))], 8, 0),
        kind(
            "model shrinkage",
            vec![("model_shrink_rate", json!(0.1))],
            8,
            0,
        ),
    ];
    for kind in &kinds {
        let (model, x) = model(kind);
        let (name, n_cols) = (kind.name, kind.n_cols);
        let row = &x[n_cols..2 * n_cols];
        let mut margins = vec![0.0; model.n_outputs()];
        let mut values = vec![0.0; model.prediction_width()];
        // The first prediction lays out the trees and the transform.
        model
            .predict_row_into(row, Iterations::Best, &mut values)
            .unwrap();
        let single = model.n_outputs() == 1;
        let count = allocations(|| {
            for _ in 0..3 {
                model
                    .predict_margin_row_into(row, Iterations::Best, &mut margins)
                    .unwrap();
                model
                    .predict_row_into(row, Iterations::Best, &mut values)
                    .unwrap();
                model.transform_margins_into(&margins, &mut values).unwrap();
                if single {
                    let margin = model.predict_margin_row(row, ..).unwrap();
                    model.predict_row(row, ..).unwrap();
                    model.transform_margin(margin).unwrap();
                }
            }
        });
        assert_eq!(count, 0, "{name}");
    }
}
