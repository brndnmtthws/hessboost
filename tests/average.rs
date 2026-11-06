//! `BoostedModel::average`: the bagged model's trees, leaves and intercepts
//! follow its documented arithmetic, its margins are the members' mean, and
//! members that do not share a margin layout are refused. Its trees record
//! the member count as their leaf divisor, which every format keeps and
//! which refresh (`process_type=update`) divides refreshed leaves by.

mod common;

use common::{four_features, incompatible_model, invalid_param, labeled_dense};
use hessboost::config::{
    BoosterKind, Boulevard, Dart, Ebm, LinearTree, MaxDeltaStep, ModelShrink, ModelShrinkMode,
    MultiStrategy, ProcessType, Refresh, TrainingParamsBuilder,
};
use hessboost::data::FeatureType;
use hessboost::objective::{Multiclass, Objective, Quantiles};
use hessboost::prelude::*;
use hessboost::tree::RegTree;
use serde_json::Value;
use std::num::NonZeroUsize;

const ROWS: usize = 300;

/// `ROWS` rows of [`four_features`] with a smooth label, keeping every row
/// `i` with `keep(i)` (members trained on different rows get different
/// intercepts).
fn data_where(keep: impl Fn(usize) -> bool) -> DMatrix {
    let rows: Vec<usize> = (0..ROWS).filter(|&i| keep(i)).collect();
    let x: Vec<f32> = rows.iter().flat_map(|&i| four_features(i)).collect();
    let y: Vec<f32> = rows
        .iter()
        .map(|&i| {
            let [a, b, c, _] = four_features(i);
            3.0 * a - 2.0 * b * c + (i % 5) as f32 * 0.1
        })
        .collect();
    labeled_dense(&x, 4, &y)
}

fn data() -> DMatrix {
    data_where(|_| true)
}

fn base() -> TrainingParamsBuilder {
    TrainingParams::builder().max_depth(3).subsample(0.7)
}

/// Members differing in seed and rows, so trees and intercepts differ.
fn bag(params: impl Fn(u64) -> TrainingParamsBuilder) -> Vec<BoostedModel> {
    (0..3u64)
        .map(|seed| {
            let rows = data_where(|i| !(i as u64 + seed).is_multiple_of(4));
            train(&params(seed).seed(seed).build().unwrap(), &rows, 8).unwrap()
        })
        .collect()
}

fn margins(model: &BoostedModel) -> Vec<f32> {
    model
        .predict_margin(&data(), Iterations::Best)
        .unwrap()
        .into_vec()
}

/// Every leaf of the average is the member leaf `v` as the `f32` nearest
/// `v / 3`, the trees come in member order, and each intercept is the
/// members' `f64` mean rounded to `f32`; the margins are the members' mean.
#[test]
fn average_divides_leaves_and_averages_intercepts() {
    let members = bag(|_| base());
    let average = BoostedModel::average(&members).unwrap();
    let trees: Vec<_> = members.iter().flat_map(BoostedModel::trees).collect();
    assert_eq!(average.trees().len(), trees.len());
    assert_eq!(average.num_boost_rounds(), 24);
    assert_eq!(average.best_iteration(), None);
    for (got, member) in average.trees().iter().zip(trees) {
        assert_eq!(got.num_nodes(), member.num_nodes());
        for node in 0..member.num_nodes() {
            let (got, member) = (got.node(node), member.node(node));
            let expected = if member.is_leaf() {
                (f64::from(member.leaf_value) / 3.0) as f32
            } else {
                member.leaf_value
            };
            assert_eq!(got.leaf_value.to_bits(), expected.to_bits());
            assert_eq!(got.sum_hess.to_bits(), member.sum_hess.to_bits());
        }
    }
    let intercepts: Vec<f32> = members.iter().map(BoostedModel::base_score).collect();
    assert!(intercepts[0] != intercepts[1], "{intercepts:?}");
    let mean = (intercepts.iter().map(|&b| f64::from(b)).sum::<f64>() / 3.0) as f32;
    assert_eq!(average.base_score().to_bits(), mean.to_bits());

    let member_margins: Vec<Vec<f32>> = members.iter().map(margins).collect();
    for (row, got) in margins(&average).into_iter().enumerate() {
        let expected = member_margins
            .iter()
            .map(|m| f64::from(m[row]))
            .sum::<f64>()
            / 3.0;
        assert!((f64::from(got) - expected).abs() < 1e-5, "row {row}");
    }
}

/// Averaging one model reproduces it bit for bit, and members with equal
/// intercepts keep that intercept exactly.
#[test]
fn a_single_member_is_its_own_average() {
    let model = &bag(|_| base())[0];
    let average = BoostedModel::average([model]).unwrap();
    assert_eq!(margins(&average), margins(model));
    let twice = BoostedModel::average([model, model]).unwrap();
    assert_eq!(twice.base_score().to_bits(), model.base_score().to_bits());
}

/// Vector leaves, linear leaves and alpha lists average output by output:
/// the margins are the members' mean in every output.
#[test]
fn every_leaf_kind_and_output_is_averaged() {
    let quantiles = Objective::Quantile(Quantiles::new(vec![0.2, 0.8]).unwrap());
    let cases = [
        bag(|_| base().linear_tree(LinearTree::default())),
        bag(|_| base().objective(quantiles.clone())),
        bag(|_| {
            base()
                .objective(quantiles.clone())
                .multi_strategy(MultiStrategy::MultiOutputTree)
        }),
    ];
    for members in cases {
        let average = BoostedModel::average(&members).unwrap();
        let member_margins: Vec<Vec<f32>> = members.iter().map(margins).collect();
        for (cell, got) in margins(&average).into_iter().enumerate() {
            let expected = member_margins
                .iter()
                .map(|m| f64::from(m[cell]))
                .sum::<f64>()
                / 3.0;
            assert!((f64::from(got) - expected).abs() < 1e-5, "cell {cell}");
        }
    }
}

/// Members whose margins do not combine by concatenation are refused.
#[test]
fn incompatible_members_are_refused() {
    assert_eq!(
        invalid_param(BoostedModel::average(std::iter::empty())),
        "models"
    );
    let plain = &bag(|_| base())[0];
    let train_with =
        |params: TrainingParamsBuilder| train(&params.build().unwrap(), &data(), 4).unwrap();
    let trained_on = |data: &DMatrix| train(&base().build().unwrap(), data, 4).unwrap();
    let wider = {
        let x: Vec<f32> = (0..ROWS).flat_map(|i| [four_features(i)[0]; 5]).collect();
        let y: Vec<f32> = (0..ROWS).map(|i| four_features(i)[0]).collect();
        trained_on(&labeled_dense(&x, 5, &y))
    };
    let early_stopped = {
        let data = data();
        Trainer::new(&base().build().unwrap(), &data, 50)
            .eval(&data, "train")
            .early_stopping_rounds(NonZeroUsize::new(1).unwrap())
            .train()
            .unwrap()
            .model
    };
    let refused = [
        train_with(base().objective(Objective::AbsoluteError)),
        train_with(base().objective(Objective::Quantile(Quantiles::new(vec![0.3]).unwrap()))),
        wider,
        train_with(base().num_parallel_tree(2)),
        train_with(base().booster(BoosterKind::Dart(
            Dart::builder().rate_drop(0.5).build().unwrap(),
        ))),
        train_with(TrainingParams::builder().booster(BoosterKind::GbLinear)),
        train_with(base().model_shrink(ModelShrink::new(0.3, ModelShrinkMode::Constant).unwrap())),
        train_with(
            TrainingParams::builder()
                .booster(BoosterKind::Boulevard(
                    Boulevard::builder().build().unwrap(),
                ))
                .eta(0.8)
                .subsample(0.8),
        ),
        train_with(
            TrainingParams::builder().booster(BoosterKind::Ebm(Ebm::builder().build().unwrap())),
        ),
        early_stopped,
    ];
    for (case, member) in refused.iter().enumerate() {
        assert_eq!(
            incompatible_model(BoostedModel::average([plain, member])),
            "average",
            "case {case}"
        );
    }
}

/// Members with categorical splits keep their category sets: the average's
/// margins are the members' mean.
#[test]
fn categorical_members_average() {
    let x: Vec<f32> = (0..ROWS)
        .flat_map(|i| [(i % 6) as f32, four_features(i)[0]])
        .collect();
    let y: Vec<f32> = (0..ROWS)
        .map(|i| [3.0, -1.0, 0.5, 2.0, -2.0, 1.0][i % 6] + four_features(i)[0])
        .collect();
    let data = labeled_dense(&x, 2, &y)
        .with_feature_types(&[FeatureType::Categorical, FeatureType::Numerical])
        .unwrap();
    let members: Vec<BoostedModel> = (0..3u64)
        .map(|seed| train(&base().seed(seed).build().unwrap(), &data, 6).unwrap())
        .collect();
    assert!(
        members[0]
            .trees()
            .iter()
            .any(|tree| tree.nodes().iter().any(|node| node.is_categorical))
    );
    let average = BoostedModel::average(&members).unwrap();
    let margins = |m: &BoostedModel| {
        m.predict_margin(&data, Iterations::Best)
            .unwrap()
            .into_vec()
    };
    let member_margins: Vec<Vec<f32>> = members.iter().map(margins).collect();
    for (row, got) in margins(&average).into_iter().enumerate() {
        let expected = member_margins
            .iter()
            .map(|m| f64::from(m[row]))
            .sum::<f64>()
            / 3.0;
        assert!((f64::from(got) - expected).abs() < 1e-5, "row {row}");
    }
}

/// Members that differ only in a stored setting or class count, or in leaf
/// kind (also behind a tree-less first member), are refused.
#[test]
fn members_differing_in_layout_are_refused() {
    let data = data();
    let plain = train(&base().build().unwrap(), &data, 3).unwrap();
    let bounded = train(
        &base()
            .max_delta_step(MaxDeltaStep::Bounded(0.5))
            .build()
            .unwrap(),
        &data,
        3,
    )
    .unwrap();
    assert_eq!(
        incompatible_model(BoostedModel::average([&plain, &bounded])),
        "average"
    );

    let classes = |k: usize| {
        let x: Vec<f32> = (0..ROWS).flat_map(four_features).collect();
        let y: Vec<f32> = (0..ROWS).map(|i| (i % k) as f32).collect();
        let params = base()
            .objective(Objective::Softprob(Multiclass::new(k).unwrap()))
            .build()
            .unwrap();
        train(&params, &labeled_dense(&x, 4, &y), 2).unwrap()
    };
    assert_eq!(
        incompatible_model(BoostedModel::average([&classes(3), &classes(4)])),
        "average"
    );

    let quantiles =
        || base().objective(Objective::Quantile(Quantiles::new(vec![0.2, 0.8]).unwrap()));
    let empty = train(&quantiles().build().unwrap(), &data, 0).unwrap();
    let scalar = train(&quantiles().build().unwrap(), &data, 2).unwrap();
    let vector = train(
        &quantiles()
            .multi_strategy(MultiStrategy::MultiOutputTree)
            .build()
            .unwrap(),
        &data,
        2,
    )
    .unwrap();
    assert_eq!(empty.num_trees(), 0);
    for members in [[&scalar, &vector, &empty], [&empty, &vector, &scalar]] {
        assert_eq!(
            incompatible_model(BoostedModel::average(members)),
            "average"
        );
    }
    // A tree-less member fits either kind.
    let average = BoostedModel::average([&empty, &vector]).unwrap();
    assert!(average.has_vector_leaves());
    average.predict_margin(&data, Iterations::Best).unwrap();
}

/// Rows to refresh on: other rows of [`four_features`] with labels `label`
/// of their first three features (more than one refresh statistics block).
fn refresh_rows(label: fn([f32; 4]) -> f32) -> DMatrix {
    let n = 9_000;
    let row = |i: usize| four_features(i * 3 + 1);
    let x: Vec<f32> = (0..n).flat_map(row).collect();
    let y: Vec<f32> = (0..n).map(|i| label(row(i))).collect();
    labeled_dense(&x, 4, &y)
}

/// `model` refreshed (`process_type=update` with `refresh_leaf`) on `rows`
/// for `rounds` iterations with `params`.
fn refreshed(
    model: &BoostedModel,
    params: TrainingParamsBuilder,
    rows: &DMatrix,
    rounds: usize,
) -> BoostedModel {
    let params = params
        .process_type(ProcessType::Update(Refresh::default()))
        .build()
        .unwrap();
    Trainer::new(&params, rows, rounds)
        .init_model(model)
        .train()
        .unwrap()
        .model
}

fn divisors(model: &BoostedModel) -> Vec<u64> {
    model.trees().iter().map(RegTree::leaf_divisor).collect()
}

/// The acceptance property of refreshing an average: for members fit on
/// the same data (differing in seed), refreshing their average on rows `R`
/// gives, bit for bit, each member tree refreshed with the same `eta`
/// against the average's own gradients on `R` (those of the already
/// refreshed iterations, as refresh computes them) with its leaves divided
/// by the member count as `average` divides them. Covers, gains and the
/// divisor are kept, at any thread count; a one-member average (divisor 1)
/// refreshes exactly as its member does.
#[test]
fn refreshing_an_average_refreshes_each_members_share() {
    type Case = (Objective, usize, fn([f32; 4]) -> f32);
    let cases: [Case; 2] = [
        (
            Objective::SquaredError(RegLoss::default()),
            1,
            |[a, b, c, _]| 2.0 * a - b * c + 1.5,
        ),
        (
            Objective::BinaryLogistic(RegLoss::default()),
            2,
            |[a, b, _, _]| f32::from(u8::from(a + b > 1.0)),
        ),
    ];
    for (objective, parallel, label) in cases {
        let params = || {
            TrainingParams::builder()
                .objective(objective.clone())
                .max_depth(3)
                .eta(0.3)
                .num_parallel_tree(parallel)
        };
        let rows = refresh_rows(label);
        let fit = refresh_rows(|[a, b, c, _]| f32::from(u8::from(a * b + c > 0.6)));
        let members: Vec<BoostedModel> = (0..3u64)
            .map(|seed| {
                let sampled = params().subsample(0.7).colsample_bynode(0.75).seed(seed);
                train(&sampled.build().unwrap(), &fit, 4).unwrap()
            })
            .collect();
        let average = BoostedModel::average(&members).unwrap();
        assert_eq!(divisors(&average), vec![3; 12 * parallel]);
        let got = refreshed(&average, params().nthread(1), &rows, 12);
        let threaded = refreshed(&average, params().nthread(8), &rows, 12);
        assert_eq!(
            got.encode(ModelFormat::Binary).unwrap(),
            threaded.encode(ModelFormat::Binary).unwrap()
        );
        assert_eq!(divisors(&got), divisors(&average));

        let n = rows.n_rows();
        for iteration in 0..12 {
            // The average's own margins on `R` before this iteration.
            let margins = if iteration == 0 {
                vec![got.base_score(); n]
            } else {
                got.slice(..iteration, 1)
                    .unwrap()
                    .predict_margin(&rows, Iterations::Best)
                    .unwrap()
                    .into_vec()
            };
            let (member, own) = (iteration / 4, iteration % 4);
            let trees = refreshed(
                &members[member].slice(own..=own, 1).unwrap(),
                params(),
                &rows.clone().with_base_margin(&margins).unwrap(),
                1,
            );
            let first = iteration * parallel;
            for (slot, expected) in trees.trees().iter().enumerate() {
                let tree = &got.trees()[first + slot];
                assert_eq!(expected.leaf_divisor(), 1);
                assert_eq!(tree.num_nodes(), expected.num_nodes());
                for (node, (got, expected)) in tree.nodes().iter().zip(expected.nodes()).enumerate()
                {
                    let leaf = if expected.is_leaf() {
                        (f64::from(expected.leaf_value) / 3.0) as f32
                    } else {
                        expected.leaf_value
                    };
                    let what =
                        format!("{objective:?}: iteration {iteration}, slot {slot}, node {node}");
                    assert_eq!(got.leaf_value.to_bits(), leaf.to_bits(), "{what}");
                    assert_eq!(
                        got.sum_hess.to_bits(),
                        expected.sum_hess.to_bits(),
                        "{what}"
                    );
                    assert_eq!(
                        got.split_gain.to_bits(),
                        expected.split_gain.to_bits(),
                        "{what}"
                    );
                }
            }
        }

        // Refreshing statistics only keeps the averaged leaves and divisors.
        let stats_only = params()
            .process_type(ProcessType::Update(Refresh::stats_only()))
            .build()
            .unwrap();
        let kept = Trainer::new(&stats_only, &rows, 12)
            .init_model(&average)
            .train()
            .unwrap()
            .model;
        assert_eq!(divisors(&kept), divisors(&average));
        for (kept, average) in kept.trees().iter().zip(average.trees()) {
            for (k, a) in kept.nodes().iter().zip(average.nodes()) {
                assert_eq!(k.leaf_value.to_bits(), a.leaf_value.to_bits());
            }
        }

        let alone = BoostedModel::average([&members[0]]).unwrap();
        assert_eq!(divisors(&alone), vec![1; 4 * parallel]);
        assert_eq!(
            refreshed(&alone, params(), &rows, 4)
                .encode(ModelFormat::Binary)
                .unwrap(),
            refreshed(&members[0], params(), &rows, 4)
                .encode(ModelFormat::Binary)
                .unwrap()
        );
    }
}

/// Every format that stores trees keeps the divisors (an average of an
/// average records the product), so a loaded average refreshes as the
/// average does; models that are not averages store none, and divisors
/// outside `1..=2^53` are refused, by loaders and by `average`.
#[test]
fn leaf_divisors_survive_every_format() {
    let members = bag(|_| base());
    let average = BoostedModel::average(&members).unwrap();
    let nested = BoostedModel::average([&average, &members[0]]).unwrap();
    assert_eq!(divisors(&nested), [vec![6; 24], vec![2; 8]].concat());
    let rows = refresh_rows(|[a, b, c, _]| a - b + c);
    let update = || TrainingParams::builder().max_depth(3);
    let expected = refreshed(&nested, update(), &rows, 32)
        .predict_margin(&rows, Iterations::Best)
        .unwrap();
    let formats = [
        ModelFormat::Binary,
        ModelFormat::Json,
        ModelFormat::XgboostJson,
        ModelFormat::XgboostUbjson,
    ];
    for format in formats {
        let loaded = BoostedModel::decode(nested.encode(format).unwrap(), format).unwrap();
        assert_eq!(divisors(&loaded), divisors(&nested), "{format:?}");
        let margins = refreshed(&loaded, update(), &rows, 32)
            .predict_margin(&rows, Iterations::Best)
            .unwrap();
        assert_eq!(margins, expected, "{format:?}");
        let plain = BoostedModel::decode(members[0].encode(format).unwrap(), format).unwrap();
        assert_eq!(divisors(&plain), vec![1; 8], "{format:?}");
    }
    let container = |model: &BoostedModel| {
        zstd::stream::decode_all(model.encode(ModelFormat::Binary).unwrap().as_slice()).unwrap()
    };
    let mentions = |bytes: &[u8], key: &str| bytes.windows(key.len()).any(|w| w == key.as_bytes());
    assert!(mentions(&container(&nested), "tree.leaf_divisor"));
    assert!(!mentions(&container(&members[0]), "tree.leaf_divisor"));
    for (format, key) in [
        (ModelFormat::Json, "leaf_divisor"),
        (ModelFormat::XgboostJson, "hessboost_leaf_divisor"),
    ] {
        assert!(mentions(&nested.encode(format).unwrap(), key));
        assert!(!mentions(&members[0].encode(format).unwrap(), key));
    }

    let doc = |format: ModelFormat| -> Value {
        serde_json::from_slice(&nested.encode(format).unwrap()).unwrap()
    };
    for bad in [0u64, (1 << 53) + 1] {
        let mut native = doc(ModelFormat::Json);
        native["trees"][0]["leaf_divisor"] = bad.into();
        assert!(BoostedModel::decode(native.to_string(), ModelFormat::Json).is_err());
        let mut xgboost = doc(ModelFormat::XgboostJson);
        xgboost["learner"]["gradient_booster"]["model"]["trees"][0]["hessboost_leaf_divisor"] =
            bad.into();
        assert!(matches!(
            BoostedModel::decode(xgboost.to_string(), ModelFormat::XgboostJson),
            Err(HessboostError::ModelFormat { .. })
        ));
    }
    let mut largest = doc(ModelFormat::Json);
    largest["trees"][0]["leaf_divisor"] = (1u64 << 53).into();
    let largest = BoostedModel::decode(largest.to_string(), ModelFormat::Json).unwrap();
    assert_eq!(largest.trees()[0].leaf_divisor(), 1 << 53);
    assert_eq!(
        incompatible_model(BoostedModel::average([&largest, &largest])),
        "average"
    );
}
