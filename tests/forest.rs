//! ForestFlow / ForestDiffusion (`hessboost::diffusion::forest`): the
//! generated rows' structure and validity, imputation, determinism,
//! persistence, and refusals.

use hessboost::config::{ProcessType, Refresh};
use hessboost::diffusion::forest::{ColumnKind, ForestMethod, ForestModel, ForestParams, Repaint};
use hessboost::prelude::*;

mod common;
use common::{invalid_param, lcg, with_threads};

const COLS: usize = 3;

/// Rows `[a, b = 2a + noise, category]` with a class label: class 1 shifts
/// `a` by 3 and always has category 7.
fn table(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut next = lcg(seed);
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for i in 0..n {
        let class = (i % 2) as f32;
        let a = next() + 3.0 * class;
        let b = 2.0 * a + 0.05 * (next() - 0.5);
        let category = if class == 1.0 || next() < 0.5 {
            7.0
        } else {
            3.0
        };
        x.extend_from_slice(&[a, b, category]);
        y.push(class);
    }
    (x, y)
}

fn quick(mut params: ForestParams) -> ForestParams {
    params.n_t = 30;
    params.duplicate_k = 20;
    params.num_boost_round = 30;
    params.column_kinds = vec![
        ColumnKind::Continuous,
        ColumnKind::Continuous,
        ColumnKind::Categorical,
    ];
    params
}

fn labelled(x: &[f32], y: &[f32]) -> DMatrix {
    DMatrix::from_dense(x, y.len(), COLS)
        .unwrap()
        .with_labels(y)
        .unwrap()
}

#[test]
fn generated_rows_follow_each_class() {
    let (x, y) = table(300, 1);
    for params in [
        quick(ForestParams::default()),
        quick(ForestParams::diffusion()),
    ] {
        let model = ForestModel::fit(&params, &labelled(&x, &y)).unwrap();
        let synthetic = model.generate(400, 3).unwrap();
        let labels = synthetic.labels().unwrap();
        let ones = labels.iter().filter(|&&l| l == 1.0).count();
        assert!(
            (150..250).contains(&ones),
            "{:?}: {ones} of 400",
            params.method
        );
        let (mut on_class, mut on_line) = (0, 0);
        for (row, &class) in synthetic.values().as_chunks::<COLS>().0.iter().zip(labels) {
            // Categories decode to seen values; every value is within range.
            assert!(row[2] == 3.0 || row[2] == 7.0);
            assert!((0.0..=4.0).contains(&row[0]), "{row:?}");
            // Most rows sit in their class's `a` range, with `b` near `2a`.
            on_class += usize::from((row[0] >= 1.5) == (class == 1.0));
            on_line += usize::from((row[1] - 2.0 * row[0]).abs() < 0.6);
        }
        assert!(on_class > 360, "{:?}: {on_class} of 400", params.method);
        assert!(on_line > 300, "{:?}: {on_line} of 400", params.method);
        let class1_cat7 = synthetic
            .values()
            .as_chunks::<COLS>()
            .0
            .iter()
            .zip(labels)
            .filter(|(r, l)| **l == 1.0 && r[2] == 7.0)
            .count();
        assert!(class1_cat7 as f64 > 0.9 * ones as f64);
        // Rows for given labels come from that class.
        let class0 = model.generate_for_labels(&[0.0; 50], 4).unwrap();
        assert!(
            class0
                .values()
                .as_chunks::<COLS>()
                .0
                .iter()
                .all(|r| r[0] < 2.0)
        );
    }
}

#[test]
fn imputation_keeps_observed_entries_and_uses_them() {
    let (x, y) = table(300, 2);
    let mut next = lcg(5);
    // Hide `b` in a third of the rows; it is a function of the observed `a`.
    let masked: Vec<f32> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if i % COLS == 1 && next() < 0.33 {
                f32::NAN
            } else {
                v
            }
        })
        .collect();
    let data = labelled(&masked, &y);
    let model = ForestModel::fit(&quick(ForestParams::diffusion()), &data).unwrap();
    for repaint in [None, Some(Repaint::default())] {
        let imputed = model.impute(&data, 2, repaint, 1).unwrap();
        assert_eq!(imputed.len(), 2 * 300 * COLS);
        let first = &imputed[..300 * COLS];
        let (mut se, mut holes) = (0.0, 0);
        for ((m, i), t) in masked.iter().zip(first).zip(&x) {
            if m.is_nan() {
                assert!(i.is_finite());
                se += f64::from(i - t).powi(2);
                holes += 1;
            } else {
                assert_eq!(m, i);
            }
        }
        let rmse = (se / f64::from(holes)).sqrt();
        // `b` spans about [0, 2] ∪ [6, 8]; ignoring `a` and the class (the
        // mean) would be off by ~3.
        assert!(rmse < 1.5, "{repaint:?}: RMSE {rmse}");
        // The two imputations are different draws.
        assert_ne!(first, &imputed[300 * COLS..]);
    }
    // Flow matching cannot impute.
    let flow = ForestModel::fit(&quick(ForestParams::default()), &labelled(&x, &y)).unwrap();
    assert_eq!(invalid_param(flow.impute(&data, 1, None, 0)), "method");
}

#[test]
fn fitting_and_generation_ignore_the_thread_count() {
    let (x, y) = table(120, 3);
    let run = |threads| {
        with_threads(threads, || {
            let model =
                ForestModel::fit(&quick(ForestParams::diffusion()), &labelled(&x, &y)).unwrap();
            model.generate(30, 9).unwrap()
        })
    };
    let one = run(1);
    assert_eq!(one, run(4));
    // Rows depend on their index only: fewer rows are a prefix.
    let model = ForestModel::fit(&quick(ForestParams::diffusion()), &labelled(&x, &y)).unwrap();
    let fewer = model.generate(10, 9).unwrap();
    assert_eq!(fewer.values(), &one.values()[..10 * COLS]);
}

#[test]
fn both_formats_round_trip() {
    let (x, y) = table(120, 4);
    let mut masked = x.clone();
    masked[4] = f32::NAN;
    for (params, data) in [
        (quick(ForestParams::default()), labelled(&x, &y)),
        // Missing values: one GBDT per column.
        (quick(ForestParams::diffusion()), labelled(&masked, &y)),
        // Unconditional.
        (
            quick(ForestParams::default()),
            DMatrix::from_dense(&x, 120, COLS).unwrap(),
        ),
    ] {
        let model = ForestModel::fit(&params, &data).unwrap();
        let expected = model.generate(20, 1).unwrap();
        let bytes = model.to_bytes().unwrap();
        for loaded in [
            ForestModel::from_bytes(&bytes).unwrap(),
            ForestModel::from_json(&model.to_json().unwrap()).unwrap(),
        ] {
            assert_eq!(loaded.method(), model.method());
            assert_eq!(loaded.generate(20, 1).unwrap(), expected);
        }
        assert!(matches!(
            ForestModel::from_bytes(&bytes[..bytes.len() / 2]),
            Err(HessboostError::ModelFormat(_))
        ));
    }
}

#[test]
fn unsupported_inputs_are_refused() {
    let (x, y) = table(60, 6);
    let data = labelled(&x, &y);
    let mut params = quick(ForestParams::default());
    params.n_t = 1;
    assert_eq!(invalid_param(ForestModel::fit(&params, &data)), "n_t");
    let mut params = quick(ForestParams::default());
    params.method = ForestMethod::Diffusion {
        beta_min: 1.0,
        beta_max: 0.5,
    };
    assert_eq!(invalid_param(ForestModel::fit(&params, &data)), "method");
    let mut params = quick(ForestParams::default());
    params.column_kinds.pop();
    assert!(matches!(
        ForestModel::fit(&params, &data),
        Err(HessboostError::DimensionMismatch { .. })
    ));
    let weighted = labelled(&x, &y).with_weights(&[1.0; 60]).unwrap();
    assert_eq!(
        invalid_param(ForestModel::fit(&quick(ForestParams::default()), &weighted)),
        "data"
    );

    let model = ForestModel::fit(&quick(ForestParams::diffusion()), &data).unwrap();
    assert_eq!(invalid_param(model.generate(0, 1)), "n_rows");
    assert_eq!(invalid_param(model.generate(usize::MAX, 1)), "n_rows");
    assert_eq!(
        invalid_param(model.generate_for_labels(&[2.0], 1)),
        "labels"
    );
    let unlabelled = DMatrix::from_dense(&x, 60, COLS).unwrap();
    assert_eq!(invalid_param(model.impute(&unlabelled, 1, None, 1)), "data");
    let mut unseen = x.clone();
    unseen[2] = 5.0;
    assert_eq!(
        invalid_param(model.impute(&labelled(&unseen, &y), 1, None, 1)),
        "data"
    );
    assert_eq!(
        invalid_param(model.impute(&data, usize::MAX, None, 1)),
        "n_imputations"
    );
}

#[test]
fn refresh_training_params_are_refused() {
    // Every level's GBDT is trained from scratch: there is nothing to refresh.
    let (x, y) = table(60, 6);
    let mut params = quick(ForestParams::diffusion());
    params.training.process_type = ProcessType::Update(Refresh::default());
    assert_eq!(invalid_param(params.validate()), "training");
    assert_eq!(
        invalid_param(ForestModel::fit(&params, &labelled(&x, &y))),
        "training"
    );
}

#[test]
fn stored_values_outside_f32_are_refused() {
    // Labels, categories and ranges decode to `f32`: a document holding a
    // value no `f32` fit could have produced is not a model.
    let (x, y) = table(60, 7);
    let model = ForestModel::fit(&quick(ForestParams::diffusion()), &labelled(&x, &y)).unwrap();
    let json: serde_json::Value = serde_json::from_str(&model.to_json().unwrap()).unwrap();
    for (path, value) in [
        ("/classes/0", serde_json::json!(-1e100)),
        ("/classes/0", serde_json::json!(0.1)),
        ("/columns/2/categories/0", serde_json::json!(1e300)),
        ("/columns/0/max", serde_json::json!(1e39)),
    ] {
        let mut doc = json.clone();
        *doc.pointer_mut(path).unwrap() = value;
        assert!(
            ForestModel::from_json(&doc.to_string()).is_err(),
            "{path} accepted"
        );
    }
}
