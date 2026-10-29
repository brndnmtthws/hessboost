//! What-if prediction and margin bounds: every variant of a base row
//! predicts bit for bit what its materialized row predicts through the row
//! methods, whatever the model (tree weights, model shrinkage, scalar and
//! vector leaves, categorical splits, linear leaves, gblinear, softmax,
//! missing values, early stopping) and across the 64-variant chunks; the
//! margin bounds of a box hold every row in it and are exact on one row.

mod common;

use common::bits::bits;
use common::{incompatible_model, invalid_data, invalid_param, lcg};
use hessboost::data::FeatureType;
use hessboost::model::RowVariants;
use hessboost::prelude::*;
use hessboost::tree::RegTree;
use serde_json::{Value, json};

const N_COLS: usize = 6;

/// Feature 0 is categorical (codes `0..6`) unless a case is numeric.
const CATEGORICAL: usize = 0;

/// Variant counts around the 64-variant chunk boundaries.
const COUNTS: [usize; 5] = [1, 63, 64, 65, 130];

/// `n` rows of features in `[0, 4)`: every 5th value missing, feature
/// [`CATEGORICAL`] an integer code.
fn features(n: usize, seed: u64) -> Vec<f32> {
    let mut next = lcg(seed);
    (0..n * N_COLS)
        .map(|i| {
            let v = next() * 4.0;
            if i % N_COLS == CATEGORICAL {
                (v * 1.5).floor()
            } else if i % 5 == 3 {
                f32::NAN
            } else {
                v
            }
        })
        .collect()
}

/// A dense matrix of rows `x`, feature [`CATEGORICAL`] categorical unless
/// `numeric`.
fn matrix(x: &[f32], numeric: bool) -> DMatrix {
    let mut types = vec![FeatureType::Numerical; N_COLS];
    if !numeric {
        types[CATEGORICAL] = FeatureType::Categorical;
    }
    DMatrix::from_dense(x, x.len() / N_COLS, N_COLS)
        .unwrap()
        .with_feature_types(&types)
        .unwrap()
}

/// One model kind.
struct Case {
    name: &'static str,
    params: Vec<(&'static str, Value)>,
    rounds: usize,
    label: fn(&[f32]) -> f32,
    /// Every feature numeric (gblinear refuses categorical features).
    numeric: bool,
    /// Train with an eval set and early stopping, so `Iterations::Best` is
    /// a strict prefix.
    early_stopping: bool,
}

fn signal(row: &[f32]) -> f32 {
    let x = |i: usize| if row[i].is_nan() { 1.5 } else { row[i] };
    x(1) * 0.8 - x(2) + (x(0) * 0.7).sin() + 0.3 * x(3) - 0.4 * x(4)
}

fn binary(row: &[f32]) -> f32 {
    f32::from(u8::from(signal(row) > 0.4))
}

fn positive(row: &[f32]) -> f32 {
    (signal(row) * 0.5).exp()
}

fn three_classes(row: &[f32]) -> f32 {
    (signal(row) * 0.8 + 1.5).floor().clamp(0.0, 2.0)
}

fn seventy_classes(row: &[f32]) -> f32 {
    ((signal(row) + 3.0).clamp(0.0, 6.99) * 10.0).floor()
}

fn cases() -> Vec<Case> {
    let case = |name, params: Vec<(&'static str, Value)>, label| Case {
        name,
        params,
        rounds: 10,
        label,
        numeric: false,
        early_stopping: false,
    };
    let numeric = |name, params, label| Case {
        numeric: true,
        ..case(name, params, label)
    };
    vec![
        numeric("numeric squared error", vec![], signal),
        case("categorical splits", vec![("max_depth", json!(4))], signal),
        Case {
            early_stopping: true,
            rounds: 40,
            ..case(
                "logistic, early stopping",
                vec![("objective", json!("binary:logistic")), ("eta", json!(0.9))],
                binary,
            )
        },
        case(
            "tweedie, dart",
            vec![
                ("objective", json!("reg:tweedie")),
                ("booster", json!("dart")),
                ("rate_drop", json!(0.3)),
            ],
            positive,
        ),
        case(
            "model shrinkage",
            vec![("model_shrink_rate", json!(0.05))],
            signal,
        ),
        case(
            "quantiles, one tree per output",
            vec![
                ("objective", json!("reg:quantileerror")),
                ("quantile_alpha", json!([0.1, 0.5, 0.9])),
            ],
            signal,
        ),
        case(
            "softprob, forest",
            vec![
                ("objective", json!("multi:softprob")),
                ("num_class", json!(3)),
                ("num_parallel_tree", json!(2)),
                ("subsample", json!(0.7)),
            ],
            three_classes,
        ),
        case(
            "softprob, vector leaves",
            vec![
                ("objective", json!("multi:softprob")),
                ("num_class", json!(3)),
                ("multi_strategy", json!("multi_output_tree")),
            ],
            three_classes,
        ),
        case("linear leaves", vec![("linear_tree", json!(true))], signal),
        numeric("gblinear", vec![("booster", json!("gblinear"))], signal),
        case(
            "softmax",
            vec![
                ("objective", json!("multi:softmax")),
                ("num_class", json!(3)),
            ],
            three_classes,
        ),
        Case {
            rounds: 3,
            ..case(
                "softmax, 70 classes",
                vec![
                    ("objective", json!("multi:softmax")),
                    ("num_class", json!(70)),
                    ("max_depth", json!(2)),
                ],
                seventy_classes,
            )
        },
        numeric(
            "numeric deep trees",
            vec![("max_depth", json!(10)), ("min_child_weight", json!(0))],
            signal,
        ),
    ]
}

fn train_case(case: &Case) -> BoostedModel {
    let x = features(400, 7);
    let label = |x: &[f32]| x.chunks(N_COLS).map(case.label).collect::<Vec<_>>();
    let dtrain = matrix(&x, case.numeric).with_labels(&label(&x)).unwrap();
    let params = TrainingParams::from_xgboost(case.params.clone()).unwrap();
    if case.early_stopping {
        let test = features(97, 11);
        let dvalid = matrix(&test, case.numeric)
            .with_labels(&label(&test))
            .unwrap();
        Trainer::new(&params, &dtrain, case.rounds)
            .eval(&dvalid, "valid")
            .early_stopping_rounds(std::num::NonZeroUsize::new(3).unwrap())
            .train()
            .unwrap()
            .model
    } else {
        train(&params, &dtrain, case.rounds).unwrap()
    }
}

/// The iteration selections a model's predictions accept (a shrunk model
/// only ranges from 0, gblinear only the whole model).
fn selections(model: &BoostedModel) -> Vec<Iterations> {
    if model.linear().is_some() {
        return vec![Iterations::Best, (..).into()];
    }
    let mut out = vec![Iterations::Best, (..).into(), (..2).into()];
    if model.shrinkage().is_none() {
        out.push((1..3).into());
    }
    out
}

/// `n` rows of values of the features `varying`: every 7th missing, the
/// categorical feature's an integer code.
fn variant_values(n: usize, varying: &[usize], seed: u64) -> Vec<f32> {
    let mut next = lcg(seed);
    (0..n * varying.len())
        .map(|i| {
            let v = next() * 4.0;
            if i % 7 == 2 {
                f32::NAN
            } else if varying[i % varying.len()] == CATEGORICAL {
                (v * 1.5).floor()
            } else {
                v
            }
        })
        .collect()
}

/// Variant `v`'s row: `base` with the features `varying` set to row `v` of
/// `values`.
fn variant_row(base: &[f32], varying: &[usize], values: &[f32], v: usize) -> Vec<f32> {
    let mut row = base.to_vec();
    for (j, &f) in varying.iter().enumerate() {
        row[f] = values[v * varying.len() + j];
    }
    row
}

#[test]
fn variants_predict_what_their_rows_predict() {
    let varyings: [&[usize]; 3] = [&[1], &[4, 2], &[CATEGORICAL, 3, 5]];
    for case in cases() {
        let model = train_case(&case);
        let name = case.name;
        if name == "categorical splits" {
            assert!(
                model.trees().iter().any(RegTree::has_categorical_splits),
                "{name}: no categorical split"
            );
        }
        let (k, width) = (model.n_outputs(), model.prediction_width());
        // Two base rows, each with missing values.
        let bases = features(2, 13);
        assert!(bases.iter().any(|v| v.is_nan()));
        for (b, base) in bases.chunks(N_COLS).enumerate() {
            for (s, varying) in varyings.iter().enumerate() {
                for n in COUNTS {
                    let values = variant_values(n, varying, (b * 10 + s) as u64 + n as u64);
                    let variants = RowVariants::new(base, varying, &values).unwrap();
                    assert_eq!(variants.len(), n);
                    assert!(!variants.is_empty());
                    for iterations in selections(&model) {
                        let mut margins = vec![0.0; n * k];
                        model
                            .predict_margin_variants_into(&variants, iterations, &mut margins)
                            .unwrap();
                        let mut predictions = vec![0.0; n * width];
                        model
                            .predict_variants_into(&variants, iterations, &mut predictions)
                            .unwrap();
                        let (mut row_margins, mut row_values) = (vec![0.0; k], vec![0.0; width]);
                        for v in 0..n {
                            let row = variant_row(base, varying, &values, v);
                            model
                                .predict_margin_row_into(&row, iterations, &mut row_margins)
                                .unwrap();
                            assert_eq!(
                                bits(&margins[v * k..(v + 1) * k]),
                                bits(&row_margins),
                                "{name}: margins of variant {v} of {n} over {varying:?}, \
                                 {iterations:?}"
                            );
                            model
                                .predict_row_into(&row, iterations, &mut row_values)
                                .unwrap();
                            assert_eq!(
                                bits(&predictions[v * width..(v + 1) * width]),
                                bits(&row_values),
                                "{name}: predictions of variant {v} of {n} over {varying:?}, \
                                 {iterations:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Varying every feature, a variant is any row: the variants of a test set
/// predict what the batch predicts for it.
#[test]
fn varying_every_feature_predicts_the_batch() {
    for case in cases() {
        let model = train_case(&case);
        let x = features(97, 17);
        let all: Vec<usize> = (0..N_COLS).collect();
        let base = vec![f32::NAN; N_COLS];
        let variants = RowVariants::new(&base, &all, &x).unwrap();
        let data = matrix(&x, case.numeric);
        let k = model.n_outputs();
        let mut margins = vec![0.0; 97 * k];
        model
            .predict_margin_variants_into(&variants, Iterations::Best, &mut margins)
            .unwrap();
        let batch = model.predict_margin(&data, Iterations::Best).unwrap();
        assert_eq!(bits(&margins), bits(batch.as_slice()), "{}", case.name);
    }
}

/// Numeric splits route every variant as the row walk does, on both mask
/// paths (a full chunk tests all 64 lanes at once, a short one variant by
/// variant) and whether the split feature's values are copied lane-major
/// (among the first eight varying features) or read strided (past them): a
/// value equal to the threshold goes right, the floats either side of it
/// go their own ways, and NaN takes the default direction, either way.
#[test]
fn numeric_split_masks_match_the_row_walk() {
    const COLS: usize = 12;
    let x: Vec<f32> = (0..COLS * 40).map(|i| ((i / COLS) % 20) as f32).collect();
    let y: Vec<f32> = (0..40).map(|r| f32::from(u8::from(r % 20 >= 10))).collect();
    let dtrain = DMatrix::from_dense(&x, 40, COLS)
        .unwrap()
        .with_labels(&y)
        .unwrap();
    let params = TrainingParams::from_xgboost([("max_depth", json!(1))]).unwrap();
    let trained = serde_json::to_value(train(&params, &dtrain, 1).unwrap()).unwrap();
    let threshold = 1.0f32;
    let candidates = [
        threshold,
        f32::from_bits(threshold.to_bits() - 1),
        f32::from_bits(threshold.to_bits() + 1),
        f32::NAN,
        -0.0,
        0.0,
        -3.0,
        7.0,
    ];
    let base = [3.0f32; COLS];
    let first: Vec<usize> = (0..COLS).collect();
    let last: Vec<usize> = (1..COLS).chain([0]).collect();
    for default_left in [true, false] {
        let mut doc = trained.clone();
        let nodes = &mut doc["trees"][0]["nodes"];
        assert_eq!(nodes.as_array().unwrap().len(), 3, "a stump");
        nodes[0]["split_feature"] = json!(0);
        nodes[0]["split_cond"] = json!(threshold);
        nodes[0]["default_left"] = json!(default_left);
        let (l, r) = (
            nodes[0]["left"].as_u64().unwrap() as usize,
            nodes[0]["right"].as_u64().unwrap() as usize,
        );
        nodes[l]["leaf_value"] = json!(-1.0);
        nodes[r]["leaf_value"] = json!(2.0);
        let model: BoostedModel = serde_json::from_value(doc).unwrap();
        let row_margin = |x0: f32| {
            let mut row = base;
            row[0] = x0;
            model.predict_margin_row(&row, ..).unwrap()
        };
        let (left, right) = (row_margin(-5.0), row_margin(5.0));
        assert_ne!(left, right);
        for varying in [&first, &last] {
            for n in [70, 15] {
                let values: Vec<f32> = (0..n)
                    .flat_map(move |v| {
                        varying.iter().map(move |&f| {
                            if f == 0 {
                                candidates[v % candidates.len()]
                            } else {
                                0.5
                            }
                        })
                    })
                    .collect();
                let variants = RowVariants::new(&base, varying, &values).unwrap();
                let mut margins = vec![0.0; n];
                model
                    .predict_margin_variants_into(&variants, .., &mut margins)
                    .unwrap();
                for (v, &margin) in margins.iter().enumerate() {
                    let x0 = candidates[v % candidates.len()];
                    let mut row = vec![0.5; COLS];
                    row[0] = x0;
                    let want = model.predict_margin_row(&row, ..).unwrap();
                    assert_eq!(margin.to_bits(), want.to_bits(), "variant {v} of {n}: {x0}");
                    let goes_left = if x0.is_nan() {
                        default_left
                    } else {
                        x0 < threshold
                    };
                    let side = if goes_left { left } else { right };
                    assert_eq!(margin.to_bits(), side.to_bits(), "variant {v} of {n}: {x0}");
                }
            }
        }
    }
}

fn mismatch(result: Result<()>) {
    match result {
        Err(HessboostError::DimensionMismatch { .. }) => {}
        other => panic!("expected a dimension mismatch, got {other:?}"),
    }
}

#[test]
fn malformed_variants_are_refused() {
    let base = [0.5; N_COLS];
    assert_eq!(invalid_param(RowVariants::new(&base, &[], &[])), "variants");
    assert_eq!(
        invalid_param(RowVariants::new(&base, &[1, 1], &[0.0, 0.0])),
        "variants"
    );
    match RowVariants::new(&base, &[1, 2], &[0.0; 3]) {
        Err(HessboostError::DimensionMismatch { .. }) => {}
        other => panic!("expected a dimension mismatch, got {other:?}"),
    }
    let empty = RowVariants::new(&base, &[1], &[]).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);

    let all = cases();
    let find = |name: &str| train_case(all.iter().find(|case| case.name == name).unwrap());
    // Three outputs per variant.
    let model = find("softprob, vector leaves");
    let mut out = [0.0; 6];
    model
        .predict_margin_variants_into(&empty, Iterations::Best, &mut [])
        .unwrap();
    let values = [1.0, 2.0];
    let variants = RowVariants::new(&base, &[3], &values).unwrap();
    model
        .predict_margin_variants_into(&variants, Iterations::Best, &mut out)
        .unwrap();
    mismatch(model.predict_margin_variants_into(&variants, Iterations::Best, &mut out[..5]));
    mismatch(model.predict_variants_into(&variants, Iterations::Best, &mut out[..3]));
    let short = RowVariants::new(&base[..5], &[3], &values).unwrap();
    mismatch(model.predict_margin_variants_into(&short, Iterations::Best, &mut out));

    let past = RowVariants::new(&base, &[N_COLS], &values).unwrap();
    assert_eq!(
        invalid_param(model.predict_variants_into(&past, Iterations::Best, &mut out)),
        "variants"
    );
    let mut infinite = base;
    infinite[2] = f32::INFINITY;
    let bad_base = RowVariants::new(&infinite, &[3], &values).unwrap();
    assert_eq!(
        invalid_data(model.predict_margin_variants_into(&bad_base, Iterations::Best, &mut out)),
        ("row", None)
    );
    let bad_values = [1.0, f32::NEG_INFINITY];
    let bad = RowVariants::new(&base, &[3], &bad_values).unwrap();
    assert_eq!(
        invalid_data(model.predict_variants_into(&bad, Iterations::Best, &mut out)),
        ("variants", None)
    );
    assert_eq!(
        incompatible_model(model.predict_variants_into(&variants, ..99, &mut out)),
        "iterations"
    );
    let shrunk = find("model shrinkage");
    assert_eq!(
        incompatible_model(shrunk.predict_margin_variants_into(&variants, 1..3, &mut out[..2])),
        "iterations"
    );
}

/// A random box: per feature an interval, a point, a missing feature, or a
/// side unbounded; the categorical feature (unless `numeric`) fixed or
/// missing.
fn random_box(next: &mut impl FnMut() -> f32, numeric: bool) -> (Vec<f32>, Vec<f32>) {
    let (mut lower, mut upper) = (vec![0.0; N_COLS], vec![0.0; N_COLS]);
    for f in 0..N_COLS {
        let (a, b, r) = (next() * 4.0, next() * 4.0, next());
        let (lo, hi) = (a.min(b), a.max(b));
        (lower[f], upper[f]) = if f == CATEGORICAL && !numeric {
            if r < 0.2 {
                (f32::NAN, f32::NAN)
            } else {
                let code = (a * 1.5).floor();
                (code, code)
            }
        } else if r < 0.15 {
            (f32::NAN, f32::NAN)
        } else if r < 0.3 {
            (f32::NEG_INFINITY, hi)
        } else if r < 0.45 {
            (lo, f32::INFINITY)
        } else if r < 0.5 {
            (f32::NEG_INFINITY, f32::INFINITY)
        } else if r < 0.6 {
            (a, a)
        } else {
            (lo, hi)
        };
    }
    (lower, upper)
}

/// Row `s` sampled from the box: its corners first, then the ends of its
/// unbounded sides pushed far out, then uniform values.
fn sample(lower: &[f32], upper: &[f32], s: usize, next: &mut impl FnMut() -> f32) -> Vec<f32> {
    lower
        .iter()
        .zip(upper)
        .map(|(&lo, &hi)| {
            if lo.is_nan() {
                return f32::NAN;
            }
            let lo_f = if lo.is_finite() {
                lo
            } else if hi.is_finite() {
                hi - 100.0
            } else {
                -100.0
            };
            let hi_f = if hi.is_finite() {
                hi
            } else {
                lo_f.max(4.0) + 100.0
            };
            match s {
                0 => lo_f,
                1 => hi_f,
                2 if lo == f32::NEG_INFINITY => -1e30,
                3 if hi == f32::INFINITY => 1e30,
                _ => (lo_f + (hi_f - lo_f) * next()).clamp(lo_f, hi_f),
            }
        })
        .collect()
}

#[test]
fn bounds_hold_every_row_in_the_box() {
    for case in cases() {
        let model = train_case(&case);
        if model.trees().iter().any(|t| t.linear_leaves().is_some()) {
            continue;
        }
        let name = case.name;
        let k = model.n_outputs();
        let mut next = lcg(29);
        for _ in 0..12 {
            let (lower, upper) = random_box(&mut next, case.numeric);
            for iterations in selections(&model) {
                let (mut low, mut high) = (vec![0.0; k], vec![0.0; k]);
                model
                    .margin_bounds_into(&lower, &upper, iterations, &mut low, &mut high)
                    .unwrap();
                let mut margins = vec![0.0; k];
                for s in 0..40 {
                    let row = sample(&lower, &upper, s, &mut next);
                    model
                        .predict_margin_row_into(&row, iterations, &mut margins)
                        .unwrap();
                    for o in 0..k {
                        assert!(
                            low[o] <= margins[o] && margins[o] <= high[o],
                            "{name}: output {o} of {row:?} is {}, outside [{}, {}] for \
                             {lower:?}..{upper:?}",
                            margins[o],
                            low[o],
                            high[o]
                        );
                    }
                }
            }
        }
    }
}

/// With every numeric feature unbounded, the bounds hold every row,
/// training rows included, and stay finite.
#[test]
fn unbounded_boxes_hold_every_row() {
    for case in cases() {
        let model = train_case(&case);
        if model.trees().iter().any(|t| t.linear_leaves().is_some()) {
            continue;
        }
        let name = case.name;
        let k = model.n_outputs();
        let x = features(400, 7);
        let codes = if case.numeric { 1 } else { 6 };
        for code in 0..codes {
            let mut lower = vec![f32::NEG_INFINITY; N_COLS];
            let mut upper = vec![f32::INFINITY; N_COLS];
            if !case.numeric {
                (lower[CATEGORICAL], upper[CATEGORICAL]) = (code as f32, code as f32);
            }
            let (mut low, mut high) = (vec![0.0; k], vec![0.0; k]);
            model
                .margin_bounds_into(&lower, &upper, Iterations::Best, &mut low, &mut high)
                .unwrap();
            if model.linear().is_none() {
                assert!(low.iter().chain(&high).all(|b| b.is_finite()), "{name}");
            }
            let mut margins = vec![0.0; k];
            for row in x.chunks(N_COLS) {
                if !case.numeric && row[CATEGORICAL] != code as f32 {
                    continue;
                }
                model
                    .predict_margin_row_into(row, Iterations::Best, &mut margins)
                    .unwrap();
                for o in 0..k {
                    assert!(low[o] <= margins[o] && margins[o] <= high[o], "{name}");
                }
            }
        }
    }
}

#[test]
fn a_point_box_bounds_its_row_exactly() {
    for case in cases() {
        let model = train_case(&case);
        if model.trees().iter().any(|t| t.linear_leaves().is_some()) {
            continue;
        }
        let name = case.name;
        let k = model.n_outputs();
        for row in features(30, 5).chunks(N_COLS) {
            for iterations in selections(&model) {
                let (mut low, mut high, mut margins) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
                model
                    .margin_bounds_into(row, row, iterations, &mut low, &mut high)
                    .unwrap();
                model
                    .predict_margin_row_into(row, iterations, &mut margins)
                    .unwrap();
                assert_eq!(bits(&low), bits(&margins), "{name}: lower of {row:?}");
                assert_eq!(bits(&high), bits(&margins), "{name}: upper of {row:?}");
            }
        }
    }
}

/// A term overflowing to `+inf` on one feature meets an unbounded feature's
/// `-inf`: the bound widens to infinity instead of `NaN`, while a box of one
/// row still adds exactly as its margin does.
#[test]
fn opposite_infinities_widen_the_bounds() {
    let x: Vec<f32> = (0..40).map(|i| (i % 7) as f32).collect();
    let y: Vec<f32> = x.chunks(2).map(|row| row[0] - row[1]).collect();
    let dtrain = DMatrix::from_dense(&x, 20, 2)
        .unwrap()
        .with_labels(&y)
        .unwrap();
    let params = TrainingParams::from_xgboost([("booster", json!("gblinear"))]).unwrap();
    let mut doc = serde_json::to_value(train(&params, &dtrain, 2).unwrap()).unwrap();
    doc["base_score"] = json!([0.0]);
    doc["linear"]["weights"] = json!([2.0, 1.0]);
    doc["linear"]["bias"] = json!([0.0]);
    let model: BoostedModel = serde_json::from_value(doc).unwrap();
    let (mut low, mut high) = ([0.0], [0.0]);
    let (lower, upper) = ([f32::MAX, f32::NEG_INFINITY], [f32::MAX, 0.0]);
    model
        .margin_bounds_into(&lower, &upper, .., &mut low, &mut high)
        .unwrap();
    for row in [[f32::MAX, 0.0], [f32::MAX, -1.0], [f32::MAX, -1e30]] {
        let margin = model.predict_margin_row(&row, ..).unwrap();
        assert_eq!(margin, f32::INFINITY, "{row:?}");
        assert!(
            low[0] <= margin && margin <= high[0],
            "{row:?} in [{low:?}, {high:?}]"
        );
    }
    assert_eq!(low[0], f32::NEG_INFINITY);
    assert_eq!(high[0], f32::INFINITY);
    let row = [f32::MAX, 0.0];
    model
        .margin_bounds_into(&row, &row, .., &mut low, &mut high)
        .unwrap();
    assert_eq!((low[0], high[0]), (f32::INFINITY, f32::INFINITY));
}

/// A tree far deeper than the bounds traversal recurses (a chain of tens of
/// thousands of splits, as an imported model may hold) is bounded by all of
/// its leaves instead of overflowing the stack, whichever child continues
/// the chain (the traversal recurses into left children and loops through
/// right ones, so the left-continued chain is the one that reaches the
/// cap): the bounds of boxes spanning the thresholds hold every row, and a
/// point box is still exactly its row's margin. The bounds run on a thread
/// with a small stack, which recursion through the whole chain would
/// overflow. Margins are recomputed from the public tree metadata, which
/// walks the chain without recursion.
#[test]
fn bounds_of_a_pathologically_deep_tree() {
    const DEPTH: usize = 40_000;
    let x: Vec<f32> = (0..40).map(|i| i as f32).collect();
    let dtrain = DMatrix::from_dense(&x, 40, 1)
        .unwrap()
        .with_labels(&x)
        .unwrap();
    let trained =
        serde_json::to_value(train(&TrainingParams::default(), &dtrain, 1).unwrap()).unwrap();
    let node = |left: i64, right: i64, cond: f32, value: f32| {
        json!({
            "split_feature": 0,
            "split_cond": cond,
            "default_left": true,
            "left": left,
            "right": right,
            "leaf_value": value,
            "sum_hess": 1.0,
            "split_gain": 0.0,
            "is_categorical": false,
            "cat_begin": 0,
            "cat_end": 0,
        })
    };
    let leaf_value = |i: usize| (i % 7) as f32 * 0.25 - 0.5;
    for continue_left in [false, true] {
        // Split `i` is node `2i`, its leaf node `2i + 1`, the next split
        // node `2i + 2`; the last node is a leaf. Continued right: `x < i`
        // reaches the leaf. Continued left: `x < DEPTH - i` continues.
        let mut nodes = Vec::with_capacity(2 * DEPTH + 1);
        for i in 0..DEPTH {
            let id = 2 * i as i64;
            nodes.push(if continue_left {
                node(id + 2, id + 1, (DEPTH - i) as f32, 0.0)
            } else {
                node(id + 1, id + 2, i as f32, 0.0)
            });
            nodes.push(node(-1, -1, 0.0, leaf_value(i)));
        }
        nodes.push(node(-1, -1, 0.0, 1.5));
        let mut doc = trained.clone();
        doc["trees"][0]["nodes"] = Value::Array(nodes);
        let model: BoostedModel = serde_json::from_value(doc).unwrap();
        let tree = &model.trees()[0];
        assert_eq!(tree.num_nodes(), 2 * DEPTH + 1);
        let weight = model.tree_weights().next().unwrap();
        let base = model.base_scores()[0];
        let margin = |x0: f32| {
            let leaf = tree.leaf_id_dense(&[x0], f32::NAN);
            base + weight * tree.leaf_vector(leaf)[0]
        };
        let bounds = |lower: f32, upper: f32| {
            std::thread::scope(|scope| {
                std::thread::Builder::new()
                    .stack_size(256 * 1024)
                    .spawn_scoped(scope, || {
                        let (mut low, mut high) = ([0.0], [0.0]);
                        model
                            .margin_bounds_into(&[lower], &[upper], .., &mut low, &mut high)
                            .map(|()| (low[0], high[0]))
                    })
                    .unwrap()
                    .join()
                    .unwrap()
                    .unwrap()
            })
        };
        let name = if continue_left { "left" } else { "right" };
        for (lower, upper) in [
            (f32::NEG_INFINITY, f32::INFINITY),
            (-1.0, 50_000.0),
            (100.0, f32::INFINITY),
            (-3.0, 30_000.5),
        ] {
            let (low, high) = bounds(lower, upper);
            for x0 in [
                -1e30, -2.0, -1.0, 0.0, 0.5, 1.0, 99.0, 100.0, 123.25, 29_999.0, 30_000.5,
                39_999.5, 40_000.0, 50_000.0, 1e9,
            ] {
                if x0 < lower || x0 > upper {
                    continue;
                }
                let m = margin(x0);
                assert!(low <= m && m <= high, "{name}: {x0} in [{lower}, {upper}]");
            }
        }
        for x0 in [-2.0, 0.0, 123.25, 39_999.5, 1e9, f32::NAN] {
            let (low, high) = bounds(x0, x0);
            let m = margin(x0);
            assert_eq!(
                (low.to_bits(), high.to_bits()),
                (m.to_bits(), m.to_bits()),
                "{name}: {x0}"
            );
        }
    }
}

#[test]
fn malformed_boxes_are_refused() {
    let all = cases();
    let find = |name: &str| train_case(all.iter().find(|case| case.name == name).unwrap());
    let model = find("categorical splits");
    let (mut low, mut high) = ([0.0], [0.0]);
    let lower = [1.0, 0.0, 0.0, f32::NAN, 0.0, 0.0];
    let upper = [1.0, 1.0, 2.0, f32::NAN, 3.0, 1.0];
    model
        .margin_bounds_into(&lower, &upper, Iterations::Best, &mut low, &mut high)
        .unwrap();
    mismatch(model.margin_bounds_into(&lower[..5], &upper, Iterations::Best, &mut low, &mut high));
    mismatch(model.margin_bounds_into(&lower, &upper[..5], Iterations::Best, &mut low, &mut high));
    mismatch(model.margin_bounds_into(&lower, &upper, Iterations::Best, &mut [], &mut high));
    mismatch(model.margin_bounds_into(&lower, &upper, Iterations::Best, &mut low, &mut [0.0; 2]));
    let refused = |l: &[f32], u: &[f32]| {
        let (mut low, mut high) = ([0.0], [0.0]);
        invalid_param(model.margin_bounds_into(l, u, Iterations::Best, &mut low, &mut high))
    };
    let with = |bounds: &[f32], f: usize, v: f32| {
        let mut b = bounds.to_vec();
        b[f] = v;
        b
    };
    // NaN in one bound only.
    assert_eq!(refused(&with(&lower, 1, f32::NAN), &upper), "bounds");
    assert_eq!(refused(&lower, &with(&upper, 2, f32::NAN)), "bounds");
    // Inverted, or holding no finite value.
    assert_eq!(refused(&with(&lower, 2, 2.5), &upper), "bounds");
    assert_eq!(
        refused(
            &with(&lower, 4, f32::INFINITY),
            &with(&upper, 4, f32::INFINITY)
        ),
        "bounds"
    );
    assert_eq!(
        refused(
            &with(&lower, 4, f32::NEG_INFINITY),
            &with(&upper, 4, f32::NEG_INFINITY)
        ),
        "bounds"
    );
    // An interval on the categorical feature.
    assert_eq!(refused(&lower, &with(&upper, CATEGORICAL, 3.0)), "bounds");
    assert_eq!(
        incompatible_model(model.margin_bounds_into(&lower, &upper, ..99, &mut low, &mut high)),
        "iterations"
    );

    let linear = find("linear leaves");
    assert_eq!(
        incompatible_model(linear.margin_bounds_into(
            &lower,
            &upper,
            Iterations::Best,
            &mut low,
            &mut high
        )),
        "linear_tree"
    );
    let shrunk = find("model shrinkage");
    assert_eq!(
        incompatible_model(shrunk.margin_bounds_into(&lower, &upper, 1..3, &mut low, &mut high)),
        "iterations"
    );
}
