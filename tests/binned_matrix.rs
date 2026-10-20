//! Binned matrices: training on a `BinnedMatrix` is training on the
//! `DMatrix` it was binned from, bit for bit, for every option it accepts,
//! and everything that would read the values it dropped is refused.

use hessboost::data::{BinnedMatrix, FeatureType, ViewCuts};
use hessboost::model::ModelFormat;
use hessboost::prelude::{DMatrix, Trainer, TrainingParams};
use hessboost::training::TrainingData;
use serde_json::{Value, json};

mod common;
use common::{four_features, incompatible_model, invalid_data, invalid_param, lcg};

const ROWS: usize = 1_200;
const MAX_BIN: usize = 64;

/// Four numeric features (one with missing values) and a categorical fifth,
/// a noisy label, and positive weights when `weighted`.
fn dense(weighted: bool) -> DMatrix {
    let mut noise = lcg(9);
    let x: Vec<f32> = (0..ROWS)
        .flat_map(|i| {
            let [a, b, c, d] = four_features(i);
            [a, b, c, d, (i % 6) as f32]
        })
        .collect();
    let y: Vec<f32> = (0..ROWS)
        .map(|i| {
            let [a, b, c, _] = four_features(i);
            let shift = if i % 6 < 3 { 1.0 } else { -0.5 };
            2.0 * a - b + c * c + shift + noise()
        })
        .collect();
    let mut types = vec![FeatureType::Numerical; 4];
    types.push(FeatureType::Categorical);
    let data = DMatrix::from_dense(&x, ROWS, 5)
        .unwrap()
        .with_labels(&y)
        .unwrap()
        .with_feature_types(&types)
        .unwrap();
    if !weighted {
        return data;
    }
    let weights: Vec<f32> = (0..ROWS).map(|i| 0.5 + (i % 4) as f32).collect();
    data.with_weights(&weights).unwrap()
}

/// Six numeric features, each present on about a third of the rows: too
/// empty for the index to keep feature columns.
fn sparse() -> DMatrix {
    let mut draw = lcg(4);
    let (mut indptr, mut indices, mut values, mut y) =
        (vec![0], Vec::new(), Vec::new(), Vec::new());
    for _ in 0..ROWS {
        let mut label = 0.0;
        for f in 0..6u32 {
            if draw() < 0.3 {
                let v = draw();
                indices.push(f);
                values.push(v);
                label += v * (f as f32 - 2.5);
            }
        }
        indptr.push(values.len());
        y.push(label + draw() * 0.1);
    }
    DMatrix::from_csr(indptr, indices, values, 6)
        .unwrap()
        .with_labels(&y)
        .unwrap()
}

/// The parameters `keys` set over a small base, which they override.
fn params(keys: &[(&str, Value)]) -> TrainingParams {
    let base = [
        ("max_depth", json!(4)),
        ("eta", json!(0.3)),
        ("max_bin", json!(MAX_BIN)),
        ("seed", json!(5)),
    ];
    let unset = base
        .into_iter()
        .filter(|(key, _)| keys.iter().all(|(set, _)| set != key));
    TrainingParams::from_xgboost(unset.chain(keys.iter().cloned())).unwrap()
}

/// The model and the eval history of 10 rounds on `dtrain` scored on
/// `deval`, stopping early after 3 rounds without improvement.
fn fit<'a>(
    params: &'a TrainingParams,
    dtrain: impl Into<TrainingData<'a>>,
    deval: &'a DMatrix,
) -> (Vec<u8>, Vec<u64>) {
    let result = Trainer::new(params, dtrain, 10)
        .eval(deval, "eval")
        .early_stopping_rounds(3.try_into().unwrap())
        .train()
        .unwrap();
    let scores = result
        .history
        .rounds()
        .flat_map(|round| {
            round
                .values()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        })
        .collect();
    (result.model.encode(ModelFormat::Binary).unwrap(), scores)
}

/// Every option binned training accepts gives the matrix's own model and
/// eval history: sampled rows route by bins, exact and vector leaves
/// included, on feature columns and on a sparse index's row bins.
#[test]
fn training_on_bins_is_training_on_their_matrix() {
    let cases: &[(&str, &[(&str, Value)])] = &[
        ("depthwise", &[]),
        (
            "sampled rows and columns",
            &[("subsample", json!(0.6)), ("colsample_bynode", json!(0.7))],
        ),
        (
            "lossguide",
            &[
                ("grow_policy", json!("lossguide")),
                ("max_leaves", json!(9)),
                ("subsample", json!(0.8)),
            ],
        ),
        (
            "forest",
            &[("num_parallel_tree", json!(3)), ("subsample", json!(0.5))],
        ),
        (
            "gradient-based sampling",
            &[
                ("sampling_method", json!("gradient_based")),
                ("subsample", json!(0.4)),
            ],
        ),
        (
            "quantized gradients",
            &[
                ("use_quantized_grad", json!(true)),
                ("quant_train_renew_leaf", json!(true)),
            ],
        ),
        ("extra trees", &[("extra_trees", json!(true))]),
        ("path smoothing", &[("path_smooth", json!(2.0))]),
        (
            "constraints",
            &[
                ("monotone_constraints", json!([1, 0, 0, 0, 0])),
                ("interaction_constraints", json!([[0, 1], [2, 3, 4]])),
            ],
        ),
        (
            "exact quantile leaves",
            &[
                ("objective", json!("reg:quantileerror")),
                ("quantile_alpha", json!(0.3)),
                ("leaf_estimation_method", json!("exact")),
                ("subsample", json!(0.7)),
            ],
        ),
        (
            "vector leaves",
            &[
                ("objective", json!("reg:quantileerror")),
                ("quantile_alpha", json!([0.2, 0.8])),
                ("multi_strategy", json!("multi_output_tree")),
                ("subsample", json!(0.7)),
            ],
        ),
    ];
    let data = dense(true);
    let binned = BinnedMatrix::new(&data, MAX_BIN).unwrap();
    for (name, keys) in cases {
        let params = params(keys);
        assert_eq!(
            fit(&params, &binned, &data),
            fit(&params, &data, &data),
            "{name}"
        );
    }
    let data = sparse();
    let binned = BinnedMatrix::new(&data, MAX_BIN).unwrap();
    for (name, keys) in [&cases[0], &cases[1], &cases[9]].into_iter().chain([&(
        "symmetric",
        &[
            ("grow_policy", json!("symmetric")),
            ("subsample", json!(0.7)),
        ][..],
    )]) {
        let params = params(keys);
        assert_eq!(
            fit(&params, &binned, &data),
            fit(&params, &data, &data),
            "sparse {name}"
        );
    }
}

/// A binned matrix keeps its bins under new labels and query groups, and
/// trains as its matrix with them.
#[test]
fn binned_matrices_take_new_metadata() {
    let data = dense(true);
    let binned = BinnedMatrix::new(&data, MAX_BIN).unwrap();
    let classes: Vec<f32> = (0..ROWS).map(|i| ((i * 7) % 3) as f32).collect();
    let multiclass = params(&[
        ("objective", json!("multi:softprob")),
        ("num_class", json!(3)),
        ("subsample", json!(0.8)),
    ]);
    let relabeled = binned.with_labels(&classes).unwrap();
    let relabeled_data = data.with_labels(&classes).unwrap();
    assert_eq!(
        fit(&multiclass, &relabeled, &relabeled_data),
        fit(&multiclass, &relabeled_data, &relabeled_data)
    );
    // Query groups of 40 rows graded by the label (ranking takes one weight
    // per group, so this source has none).
    let data = dense(false);
    let binned = BinnedMatrix::new(&data, MAX_BIN).unwrap();
    let groups = vec![40; ROWS / 40];
    let ranking = params(&[("objective", json!("rank:ndcg")), ("subsample", json!(0.8))]);
    let grades: Vec<f32> = data
        .labels()
        .unwrap()
        .iter()
        .map(|y| y.clamp(0.0, 3.0).floor())
        .collect();
    let ranked = binned
        .with_labels(&grades)
        .unwrap()
        .with_group_sizes(&groups)
        .unwrap();
    let ranked_data = data
        .with_labels(&grades)
        .unwrap()
        .with_group_sizes(&groups)
        .unwrap();
    assert_eq!(
        fit(&ranking, &ranked, &ranked_data),
        fit(&ranking, &ranked_data, &ranked_data)
    );
}

/// Training refuses every option that reads feature values, a `max_bin`
/// other than the bins', `device = cuda` (allowed to pass the option check,
/// so refused on its own), and continuation.
#[test]
fn binned_training_refuses_what_reads_values() {
    let data = dense(true);
    let binned = BinnedMatrix::new(&data, MAX_BIN).unwrap();
    let refused = |keys: &[(&str, Value)]| {
        invalid_param(hessboost::prelude::train(&params(keys), &binned, 2))
    };
    assert_eq!(refused(&[("max_bin", json!(32))]), "max_bin");
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    assert_eq!(refused(&[("device", json!("cuda"))]), "device");
    for keys in [
        &[("tree_method", json!("approx"))][..],
        &[("tree_method", json!("exact"))],
        &[("booster", json!("dart"))],
        &[("booster", json!("gblinear"))],
        &[("linear_tree", json!(true))],
        &[("toad_penalty_feature", json!(0.1))],
        &[("langevin", json!(true))],
        &[("model_shrink_rate", json!(0.01))],
    ] {
        assert_eq!(refused(keys), "dtrain", "{keys:?}");
    }
    let params = params(&[]);
    let model = hessboost::prelude::train(&params, &data, 2).unwrap();
    let continued = Trainer::new(&params, &binned, 2).init_model(&model).train();
    assert_eq!(incompatible_model(continued), "init_model");
}

/// Binning needs every row in the cuts: a zero sample weight is refused,
/// in the matrix itself or in a source-cut view's source.
#[test]
fn binning_refuses_rows_the_cuts_never_saw() {
    let data = dense(true);
    let mut weights = data.weights().unwrap().to_vec();
    weights[10] = 0.0;
    let zero = data.clone().with_weights(&weights).unwrap();
    assert_eq!(
        invalid_data(BinnedMatrix::new(&zero, MAX_BIN)),
        ("weights", None)
    );
    // A source-cut view of rows the source weighs zero, weighted itself.
    let rows: Vec<usize> = (0..ROWS).collect();
    let view = zero
        .select_rows_with_cuts(&rows, ViewCuts::Source)
        .unwrap()
        .with_weights(&vec![1.0; ROWS])
        .unwrap();
    assert_eq!(
        invalid_data(BinnedMatrix::new(&view, MAX_BIN)),
        ("weights", None)
    );
    // Leaving the zero-weight row out bins.
    let weighted: Vec<usize> = (0..ROWS).filter(|&r| r != 10).collect();
    let view = zero
        .select_rows_with_cuts(&weighted, ViewCuts::Source)
        .unwrap();
    assert!(BinnedMatrix::new(&view, MAX_BIN).is_ok());
    assert_eq!(invalid_param(BinnedMatrix::new(&data, 1)), "max_bin");
}

/// A source-cut view of an unweighted source may weigh rows zero itself:
/// its source's cuts sketched them all the same, so it bins, and trains as
/// the view does, rows routed by their bins (exact leaves, sampled rounds).
#[test]
fn binned_source_cut_views_keep_their_sources_sketch() {
    let data = dense(false);
    let rows: Vec<usize> = (0..ROWS).collect();
    let mut weights = vec![1.0; ROWS];
    weights[10] = 0.0;
    let view = data
        .select_rows_with_cuts(&rows, ViewCuts::Source)
        .unwrap()
        .with_weights(&weights)
        .unwrap();
    let binned = BinnedMatrix::new(&view, MAX_BIN).unwrap();
    let params = params(&[
        ("objective", json!("reg:absoluteerror")),
        ("leaf_estimation_method", json!("exact")),
        ("subsample", json!(0.7)),
    ]);
    assert_eq!(fit(&params, &binned, &data), fit(&params, &view, &data));
}
