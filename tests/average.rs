//! `BoostedModel::average`: the bagged model's trees, leaves and intercepts
//! follow its documented arithmetic, its margins are the members' mean, and
//! members that do not share a margin layout are refused.

mod common;

use common::{four_features, incompatible_model, invalid_param, labeled_dense};
use hessboost::config::{
    BoosterKind, Boulevard, Dart, Ebm, LinearTree, ModelShrink, ModelShrinkMode, MultiStrategy,
    TrainingParamsBuilder,
};
use hessboost::objective::{Objective, Quantiles};
use hessboost::prelude::*;
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
