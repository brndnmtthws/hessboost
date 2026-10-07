//! The row methods run on the calling thread: they never enter rayon, so a
//! forked process, whose copy of rayon's global pool has no threads, still
//! predicts rows outside any pool (as the Python bindings call them). Its
//! own test binary, since it checks that the process never built rayon's
//! global pool.

use hessboost::objective::{Multiclass, Objective};
use hessboost::prelude::*;

/// Features of a row the dense scan would check in parallel blocks.
const WIDE: usize = 1 << 22;

/// Outputs of a row the prediction transform would split across threads.
const OUTPUTS: usize = 1 << 15;

#[test]
fn row_predictions_never_build_the_global_thread_pool() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let (wide, softprob) = pool.install(|| {
        let x: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let dtrain = DMatrix::from_dense(&x, 20, 2)
            .unwrap()
            .with_labels(&x[..20])
            .unwrap();
        let model = train(&TrainingParams::default(), &dtrain, 3).unwrap();
        // The same trees, read as taking `WIDE` features.
        let mut json = serde_json::to_value(&model).unwrap();
        json["n_features"] = WIDE.into();
        let wide: BoostedModel = serde_json::from_value(json).unwrap();
        let labels = [0.0, 1.0];
        let dtrain = DMatrix::from_dense(&x[..4], 2, 2)
            .unwrap()
            .with_labels(&labels)
            .unwrap();
        let params = TrainingParams::builder()
            .objective(Objective::Softprob(Multiclass::new(OUTPUTS).unwrap()))
            .build()
            .unwrap();
        (wide, train(&params, &dtrain, 0).unwrap())
    });
    let row = vec![0.5; WIDE];
    let margin = wide.predict_margin_row(&row, Iterations::Best).unwrap();
    wide.predict_row(&row, Iterations::Best).unwrap();
    wide.transform_margin(margin).unwrap();
    let mut out = vec![0.0; OUTPUTS];
    softprob
        .predict_margin_row_into(&[0.5, 1.5], Iterations::Best, &mut out)
        .unwrap();
    softprob
        .predict_row_into(&[0.5, 1.5], Iterations::Best, &mut out)
        .unwrap();
    assert!(
        rayon::ThreadPoolBuilder::new().build_global().is_ok(),
        "a row prediction used rayon's global pool"
    );
}
