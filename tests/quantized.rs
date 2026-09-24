//! Opt-in quantized-gradient training (`use_quantized_grad`, LightGBM's
//! quantized training): determinism across thread counts, quality close to
//! full-precision training, full-precision leaf renewal, and parameter
//! validation.

use hessboost::config::TrainingParamsBuilder;
use hessboost::prelude::*;

mod common;
use common::{invalid_param, labeled_dense, rmse};

const FEATURES: usize = 8;

/// SplitMix64-driven uniform variates in `[0, 1)`.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// A nonlinear regression task with noise; `missing` blanks some entries.
fn regression(n: usize, seed: u64, missing: bool) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Rng(seed);
    let mut x = Vec::with_capacity(n * FEATURES);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let row: Vec<f32> = (0..FEATURES).map(|_| rng.next()).collect();
        let target = 3.0 * (row[0] * 6.0).sin() + 4.0 * row[1] * row[2] - 2.0 * row[3]
            + if row[4] > 0.5 { 1.5 } else { 0.0 }
            + 0.3 * (rng.next() - 0.5);
        for (j, &v) in row.iter().enumerate() {
            x.push(if missing && rng.next() < 0.1 && j != 0 {
                f32::NAN
            } else {
                v
            });
        }
        y.push(target);
    }
    (x, y)
}

fn logloss(pred: &[f32], y: &[f32]) -> f64 {
    let total: f64 = pred
        .iter()
        .zip(y)
        .map(|(&p, &t)| {
            let p = f64::from(p).clamp(1e-7, 1.0 - 1e-7);
            -(f64::from(t) * p.ln() + (1.0 - f64::from(t)) * (1.0 - p).ln())
        })
        .sum();
    total / y.len() as f64
}

fn quantized(builder: TrainingParamsBuilder) -> TrainingParamsBuilder {
    builder.use_quantized_grad(true)
}

#[test]
fn quantized_training_is_identical_across_thread_counts() {
    let (x, y) = regression(30_000, 1, true);
    let d = labeled_dense(&x, FEATURES, &y);
    let fit = |nthread: usize, seed: u64, policy: GrowPolicy| {
        let params = quantized(TrainingParams::builder())
            .nthread(nthread)
            .seed(seed)
            .grow_policy(policy)
            .max_leaves(32)
            .max_depth(6)
            .subsample(0.8)
            .build()
            .unwrap();
        train(&params, &d, 8).unwrap().predict(&d).unwrap()
    };
    for policy in [GrowPolicy::DepthWise, GrowPolicy::LossGuide] {
        let serial = fit(1, 7, policy);
        assert_eq!(serial, fit(6, 7, policy), "{policy:?}");
        assert_eq!(serial, fit(0, 7, policy), "{policy:?}");
        // The rounding stream follows the seed.
        assert_ne!(serial, fit(1, 8, policy), "{policy:?}");
    }
}

#[test]
fn quantized_regression_stays_close_to_full_precision() {
    let (x, y) = regression(20_000, 2, false);
    let (xt, yt) = regression(5_000, 3, false);
    let (d, dt) = (
        labeled_dense(&x, FEATURES, &y),
        labeled_dense(&xt, FEATURES, &yt),
    );
    for method in [TreeMethod::Hist, TreeMethod::Approx] {
        let base = TrainingParams::builder()
            .tree_method(method)
            .max_depth(6)
            .eta(0.1);
        let score = |builder: TrainingParamsBuilder| {
            rmse(&train(&builder.build().unwrap(), &d, 150).unwrap(), &dt)
        };
        let full = score(base.clone());
        let stochastic = score(quantized(base.clone()));
        let nearest = score(quantized(base.clone()).stochastic_rounding(false));
        let renewed = score(quantized(base.clone()).quant_train_renew_leaf(true));
        let fine = score(quantized(base.clone()).num_grad_quant_bins(16));
        // Two gradient levels per sign cost a few percent; renewal and finer
        // levels close most of the gap (measured: +4.4%, +1.6%, -0.6%).
        assert!(
            stochastic < 1.06 * full,
            "{method:?}: {stochastic} vs {full}"
        );
        assert!(renewed < 1.03 * full, "{method:?}: {renewed} vs {full}");
        assert!(fine < 1.02 * full, "{method:?}: {fine} vs {full}");
        // Biased round-to-nearest is clearly worse than unbiased rounding.
        assert!(
            nearest > 1.1 * stochastic,
            "{method:?}: {nearest} vs {stochastic}"
        );
    }
}

#[test]
fn quantized_binary_classification_stays_close_to_full_precision() {
    let (x, y) = regression(20_000, 4, true);
    let (xt, yt) = regression(5_000, 5, true);
    let labels = |y: &[f32]| -> Vec<f32> { y.iter().map(|&t| f32::from(t > 1.0)).collect() };
    let (y, yt) = (labels(&y), labels(&yt));
    let (d, dt) = (
        labeled_dense(&x, FEATURES, &y),
        labeled_dense(&xt, FEATURES, &yt),
    );
    let base = TrainingParams::builder()
        .objective("binary:logistic")
        .max_depth(6)
        .eta(0.1);
    let score = |builder: TrainingParamsBuilder| {
        let model = train(&builder.build().unwrap(), &d, 150).unwrap();
        logloss(&model.predict(&dt).unwrap(), &yt)
    };
    let full = score(base.clone());
    for (name, variant) in [
        ("stochastic", quantized(base.clone())),
        (
            "renewed",
            quantized(base.clone()).quant_train_renew_leaf(true),
        ),
    ] {
        let q = score(variant);
        // Measured: +0.2% (stochastic), -3.9% (renewed).
        assert!(q < 1.02 * full, "{name}: {q} vs {full}");
    }
}

/// With renewal, every leaf holds the regularized full-precision weight of
/// the rows it received: `eta · Σy / (n + λ)` for squared error from a zero
/// margin. Without it, leaves hold the quantized weights.
#[test]
fn renewed_leaves_use_full_precision_gradients() {
    let (x, y) = regression(12_000, 6, true);
    let d = labeled_dense(&x, FEATURES, &y);
    for policy in [GrowPolicy::DepthWise, GrowPolicy::LossGuide] {
        let fit = |renew: bool| {
            let params = quantized(TrainingParams::builder())
                .quant_train_renew_leaf(renew)
                .grow_policy(policy)
                .max_leaves(24)
                .max_depth(5)
                .base_score(0.0)
                .eta(0.5)
                .build()
                .unwrap();
            train(&params, &d, 1).unwrap()
        };
        let mismatch = |model: &BoostedModel| {
            let leaves = model.predict_leaf(&d).unwrap();
            let tree = &model.trees()[0];
            let mut sums = vec![(0f64, 0usize); tree.num_nodes()];
            for (&leaf, &t) in leaves.iter().zip(&y) {
                sums[leaf as usize].0 += f64::from(t);
                sums[leaf as usize].1 += 1;
            }
            sums.iter()
                .enumerate()
                .filter(|(_, (_, n))| *n > 0)
                .map(|(leaf, &(sum, n))| {
                    let expected = 0.5 * sum / (n as f64 + 1.0);
                    (f64::from(tree.node(leaf).leaf_value) - expected).abs()
                })
                .fold(0.0, f64::max)
        };
        assert!(mismatch(&fit(true)) < 1e-5, "{policy:?}");
        assert!(mismatch(&fit(false)) > 1e-3, "{policy:?}");
    }
}

/// Subnormal gradients keep a nonzero quantized representation: rows whose
/// weights sit at the least positive `f32` still learn the unit squared-error
/// leaf `−G/H` from a zero margin, with constant and with varying Hessians.
#[test]
fn subnormal_gradients_survive_quantization() {
    let tiny = f32::from_bits(1);
    for weights in [[tiny, tiny], [tiny, 2.0 * tiny]] {
        let d = labeled_dense(&[0.0, 0.0], 1, &[1.0, 1.0])
            .with_weights(&weights)
            .unwrap();
        let params = quantized(TrainingParams::builder())
            .base_score(0.0)
            .eta(1.0)
            .lambda(0.0)
            .min_child_weight(0.0)
            .build()
            .unwrap();
        let model = train(&params, &d, 1).unwrap();
        assert_eq!(model.predict(&d).unwrap(), vec![1.0, 1.0], "{weights:?}");
    }
}

#[test]
fn quantized_parameters_are_validated() {
    for builder in [
        quantized(TrainingParams::builder()).tree_method(TreeMethod::Exact),
        quantized(TrainingParams::builder()).booster(BoosterKind::GbLinear),
        quantized(TrainingParams::builder()).multi_strategy(MultiStrategy::MultiOutputTree),
    ] {
        assert_eq!(invalid_param(builder.build()), "use_quantized_grad");
    }
    for bins in [0, 1, 128] {
        let builder = TrainingParams::builder().num_grad_quant_bins(bins);
        assert_eq!(invalid_param(builder.build()), "num_grad_quant_bins");
    }
    for bins in [2, 3, 127] {
        quantized(TrainingParams::builder())
            .num_grad_quant_bins(bins)
            .build()
            .unwrap();
    }
}
