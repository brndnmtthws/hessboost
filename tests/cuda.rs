//! CUDA backend integration tests (Linux, `cuda` feature).
//!
//! Tests that need a CUDA device skip when one is absent (CI runners have
//! no GPU) unless `HESSBOOST_REQUIRE_CUDA` is set, which turns every skip
//! into a failure: set it on a GPU machine so a broken setup cannot pass
//! vacuously. The embedded PTX itself is checked without a GPU by CI's
//! `cuda-kernels` job (rebuilt from source, assembled for every supported
//! architecture). Parameter-refusal tests always run.

#![cfg(all(target_os = "linux", feature = "cuda"))]

mod common;

use hessboost::backend::cuda::{self, CudaHistBackend, NodeCounts};
use hessboost::config::{
    BoosterKind, Dart, Device, GrowPolicy, LinearTree, MaxDeltaStep, Monotone, ProcessType,
    QuantizedGrad, Refresh, SamplingMethod, TrainingParamsBuilder,
};
use hessboost::internals::{CpuBackend, GHistIndex, HistCuts, HistogramBackend, zeroed};
use hessboost::objective::{GradPair, Multiclass, RegLoss};
use hessboost::prelude::*;
use hessboost::training::{EvalHistory, TrainResult};
use std::num::NonZeroUsize;
use std::ops::ControlFlow;
use std::sync::Barrier;

const CUDA: Device = Device::Cuda { ordinal: 0 };

/// Whether a CUDA device is usable, with the skip reason printed so a
/// vacuous pass is visible; panics instead under `HESSBOOST_REQUIRE_CUDA`.
fn device() -> bool {
    let Some(reason) = cuda::unavailable_reason() else {
        return true;
    };
    assert!(
        std::env::var_os("HESSBOOST_REQUIRE_CUDA").is_none(),
        "HESSBOOST_REQUIRE_CUDA is set but CUDA is unavailable: {reason}"
    );
    eprintln!("skipping cuda test: {reason}");
    false
}

/// The backend is either available or absent for a reason outside the
/// crate (no driver, a driver older than the backend needs, no device, a
/// device older than the kernels' target). A module load failure is never
/// an acceptable skip: without this guard, every device-dependent test
/// would pass vacuously while the backend is broken.
#[test]
fn backend_available_or_no_device() {
    if let Some(reason) = cuda::unavailable_reason() {
        assert!(
            [
                "libcuda not found",
                "the NVIDIA driver supports CUDA",
                "no CUDA device",
                "CUDA device 0 has compute capability"
            ]
            .iter()
            .any(|expected| reason.starts_with(expected)),
            "the CUDA backend failed to initialize: {reason}"
        );
    }
}

/// The unsupported `device = cuda` combinations are refused with an error,
/// never silently ignored.
#[test]
fn device_cuda_refuses_unsupported_combinations() {
    let base = TrainingParams::builder().device(CUDA).build().unwrap();
    let with = |change: fn(&mut TrainingParams)| {
        let mut params = base.clone();
        change(&mut params);
        params
    };
    let variants: Vec<(TrainingParams, &str)> = vec![
        (
            with(|p| p.tree_method = TreeMethod::Approx),
            "tree_method=approx",
        ),
        (
            with(|p| p.tree_method = TreeMethod::Exact),
            "tree_method=exact",
        ),
        (
            with(|p| p.quantized = Some(QuantizedGrad::default())),
            "use_quantized_grad",
        ),
        (
            with(|p| p.booster = BoosterKind::GbLinear),
            "booster=gblinear",
        ),
        (
            with(|p| p.process_type = ProcessType::Update(Refresh::default())),
            "process_type=update",
        ),
    ];
    for (params, name) in variants {
        assert_eq!(common::invalid_param(params.validate()), "device", "{name}");
    }
}

/// A dataset with missing values (every 13th cell) and a categorical first
/// column, as the Metal tests use.
fn dataset(n: usize, cols: usize, missing: bool) -> DMatrix {
    dataset_with(n, cols, missing, true)
}

/// [`dataset`] with the first column categorical or numeric.
fn dataset_with(n: usize, cols: usize, missing: bool, categorical: bool) -> DMatrix {
    let mut x = vec![0.0f32; n * cols];
    let mut y = vec![0.0f32; n];
    for r in 0..n {
        let mut target = 0.0;
        for f in 0..cols {
            let v = if f == 0 {
                ((r * 31 + f) % 5) as f32
            } else if missing && (r + f) % 13 == 0 {
                f32::NAN
            } else {
                (((r * 97 + f * 13) % 1000) as f32) * 0.001
            };
            x[r * cols + f] = v;
            if f > 0 && v.is_finite() {
                target += v * (f as f32);
            }
        }
        y[r] = target % 3.0;
    }
    let types: Vec<hessboost::data::FeatureType> = (0..cols)
        .map(|f| {
            if f == 0 && categorical {
                hessboost::data::FeatureType::Categorical
            } else {
                hessboost::data::FeatureType::Numerical
            }
        })
        .collect();
    DMatrix::from_dense_with_missing(&x, n, cols, f32::NAN)
        .unwrap()
        .with_feature_types(&types)
        .unwrap()
        .with_labels(&y)
        .unwrap()
}

/// The binned index of `n x cols` values from `value(row, feature)`
/// (`NaN` is missing).
fn index(n: usize, cols: usize, max_bin: usize, value: impl Fn(usize, usize) -> f32) -> GHistIndex {
    let x: Vec<f32> = (0..n * cols).map(|i| value(i / cols, i % cols)).collect();
    let data = DMatrix::from_dense_with_missing(&x, n, cols, f32::NAN).unwrap();
    let cuts = HistCuts::from_dmatrix(&data, max_bin);
    GHistIndex::from_dmatrix(&data, cuts)
}

/// The histogram of `rows` on `backend` after `prepare(gpair)`, as bits.
fn histogram(
    backend: &dyn HistogramBackend,
    index: &GHistIndex,
    rows: &[u32],
    gpair: &[GradPair],
) -> Vec<(u64, u64)> {
    let mut out = zeroed(index.total_bins());
    backend.prepare(index, gpair);
    backend.build(index, rows, gpair, &mut out);
    out.iter()
        .map(|s| (s.grad.to_bits(), s.hess.to_bits()))
        .collect()
}

/// Gradients whose sums are exact in grains (`k / 64` up to 16): every
/// node is exact.
fn exact_pairs(n: usize) -> Vec<GradPair> {
    (0..n)
        .map(|i| GradPair::new(((i * 7) % 1024) as f32 / 64.0 - 8.0, 1.0))
        .collect()
}

/// Gradients of magnitude 1 with one value of `2^-38`: `M = 2^38` grains,
/// so a chunk of up to 8,191 rows sums exactly but a node of more than
/// `2^15` rows does not.
fn chunk_exact_pairs(n: usize) -> Vec<GradPair> {
    (0..n)
        .map(|i| {
            let g = if i == 17 {
                2f32.powi(-38)
            } else if i % 3 == 0 {
                -1.0
            } else {
                1.0
            };
            GradPair::new(g, 1.0)
        })
        .collect()
}

/// Gradients spanning `1e-30` to `1e30` (and Hessians to `1e38`): no sum of
/// two of them is exact, so every node takes the `f64` chain path (or the
/// CPU's).
fn wide_pairs(n: usize) -> Vec<GradPair> {
    (0..n)
        .map(|i| {
            let e = (i * 13 % 61) as i32 - 30;
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            GradPair::new(
                sign * 10f32.powi(e) * (1.0 + (i % 7) as f32 / 8.0),
                10f32.powi((i % 39) as i32) * 0.9,
            )
        })
        .collect()
}

/// Every strategy's histograms equal the CPU backend's bit for bit: dense
/// and missing-value indexes, chains and chunked sums, inside and outside
/// the exactness domain, `u16` and `u32` bins. The node counts show each
/// strategy ran.
#[test]
fn histograms_match_cpu_for_every_strategy() {
    if !device() {
        return;
    }
    let mut seen = NodeCounts::default();
    let mut check = |index: &GHistIndex, rows: &[u32], gpair: &[GradPair], what: &str| {
        let gpu = CudaHistBackend::new(index, 0).unwrap();
        let cpu = histogram(&CpuBackend, index, rows, gpair);
        assert_eq!(histogram(&gpu, index, rows, gpair), cpu, "{what}");
        let counts = gpu.node_counts();
        if gpair
            .iter()
            .all(|p| p.grad.is_finite() && p.hess.is_finite())
        {
            assert_eq!(
                counts.cpu_nodes, 0,
                "{what}: finite histograms must run on CUDA"
            );
        }
        assert!(
            cuda::available(),
            "{what}: {:?}",
            cuda::unavailable_reason()
        );
        seen.exact_nodes += counts.exact_nodes;
        seen.exact_chunk_nodes += counts.exact_chunk_nodes;
        seen.chain_nodes += counts.chain_nodes;
        seen.cpu_nodes += counts.cpu_nodes;
    };
    let value = |r: usize, f: usize| ((r * 2_654_435_761 + f * 97) % 1009) as f32 / 7.0;
    let with_missing = |r: usize, f: usize| {
        if (r + f).is_multiple_of(5) {
            f32::NAN
        } else {
            value(r, f)
        }
    };
    let sparse = |r: usize, f: usize| {
        if (r + f).is_multiple_of(4) {
            value(r, f)
        } else {
            f32::NAN
        }
    };

    // A dense index of 2^18 rows or fewer (row subsets swept by feature): a
    // node below 8,192 rows is one chain, a larger one chunked.
    let small = index(60_000, 6, 256, value);
    let all: Vec<u32> = (0..60_000).collect();
    let thirds: Vec<u32> = (0..60_000).step_by(3).collect();
    let few: Vec<u32> = (0..60_000).step_by(11).take(5000).collect();
    check(&small, &all, &exact_pairs(60_000), "dense chain, exact");
    check(
        &small,
        &few,
        &wide_pairs(60_000),
        "dense small node, chains",
    );
    check(
        &small,
        &thirds,
        &wide_pairs(60_000),
        "dense chunked subset, chains",
    );

    // A dense index above 2^18 rows (row subsets split by rows).
    let large = index(300_000, 4, 64, value);
    let half: Vec<u32> = (0..300_000).step_by(2).collect();
    check(&large, &half, &exact_pairs(300_000), "dense chunked, exact");
    check(
        &large,
        &half,
        &chunk_exact_pairs(300_000),
        "dense chunked, exact chunks",
    );
    check(&large, &half, &wide_pairs(300_000), "dense chunked, chains");

    // Missing values (a half-full index, and a CSR-only one): chunked.
    for (name, cells) in [
        ("missing", &with_missing as &dyn Fn(usize, usize) -> f32),
        ("csr", &sparse),
    ] {
        let idx = index(40_000, 7, 128, cells);
        let rows: Vec<u32> = (0..40_000).filter(|r| r % 7 != 3).collect();
        check(&idx, &rows, &exact_pairs(40_000), &format!("{name}, exact"));
        check(
            &idx,
            &rows,
            &chunk_exact_pairs(40_000),
            &format!("{name}, exact chunks"),
        );
        check(&idx, &rows, &wide_pairs(40_000), &format!("{name}, chains"));
        check(
            &idx,
            &rows[..3000],
            &wide_pairs(40_000),
            &format!("{name}, small chains"),
        );
    }

    // More than 65,536 bins: `u32` bins.
    let wide = index(20_000, 300, 256, |r, f| ((r * 31 + f * 7) % 20_000) as f32);
    let rows: Vec<u32> = (0..20_000).collect();
    check(&wide, &rows, &exact_pairs(20_000), "u32 bins, exact");
    check(
        &wide,
        &rows[..4000],
        &wide_pairs(20_000),
        "u32 bins, chains",
    );

    // A non-finite gradient anywhere in the slice: the CPU's NaN bits.
    let mut nan = exact_pairs(60_000);
    nan[11].grad = f32::NAN;
    nan[22].hess = f32::INFINITY;
    check(&small, &few, &nan, "non-finite, cpu");

    assert!(seen.exact_nodes > 0, "{seen:?}");
    assert!(seen.exact_chunk_nodes > 0, "{seen:?}");
    assert!(seen.chain_nodes > 0, "{seen:?}");
    assert!(seen.cpu_nodes > 0, "{seen:?}");
    assert!(cuda::available(), "{:?}", cuda::unavailable_reason());
}

/// Inputs that do not fit the backend's device buffers never reach the
/// GPU: a gradient slice longer than the index and a row list longer than
/// the index give the CPU's histogram, and a row past the index is refused
/// by the CPU path's bounds check, exactly as the CPU backend refuses it.
#[test]
fn mismatched_inputs_match_the_cpu_backend() {
    if !device() {
        return;
    }
    let n = 10_000;
    let index = index(n, 1, 256, |r, _| (r % 5) as f32);
    let backend = CudaHistBackend::new(&index, 0).unwrap();
    let long: Vec<_> = (0..n + 1000)
        .map(|i| GradPair::new((i % 7) as f32 - 3.0, 1.0))
        .collect();
    let rows: Vec<u32> = (0..n as u32).collect();
    assert_eq!(
        histogram(&backend, &index, &rows, &long),
        histogram(&CpuBackend, &index, &rows, &long)
    );
    let gpair = &long[..n];
    let twice: Vec<u32> = rows.iter().chain(&rows).copied().collect();
    assert_eq!(
        histogram(&backend, &index, &twice, gpair),
        histogram(&CpuBackend, &index, &twice, gpair)
    );
    let past_end: Vec<u32> = (1..=n as u32).collect();
    let refused = |backend: &dyn HistogramBackend| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            histogram(backend, &index, &past_end, gpair)
        }))
        .is_err()
    };
    assert!(refused(&CpuBackend));
    assert!(refused(&backend));
    // Same dimensions and cut count, but different bin contents: the
    // uploaded index must not be reused, even though its shape matches.
    let other_data: Vec<f32> = (0..n).map(|r| ((r + 1) % 5) as f32).collect();
    let other = DMatrix::from_dense(&other_data, n, 1).unwrap();
    let other = GHistIndex::from_dmatrix(&other, HistCuts::from_dmatrix(&other, 256));
    let expected = histogram(&CpuBackend, &other, &rows, gpair);
    assert_ne!(expected, histogram(&CpuBackend, &index, &rows, gpair));
    assert_eq!(histogram(&backend, &other, &rows, gpair), expected);
    assert!(cuda::available(), "{:?}", cuda::unavailable_reason());
}

/// `device = cuda` training reproduces single-threaded CPU training bit for
/// bit, tree for tree (the whole serialized model compares equal), across
/// the configurations the hist builder serves.
#[test]
fn device_cuda_training_matches_single_threaded_cpu() {
    if !device() {
        return;
    }
    let dense = dataset(40_000, 10, false);
    let missing = dataset(40_000, 10, true);
    let labelled = |f: fn(f32) -> f32| {
        let y: Vec<f32> = missing.labels().unwrap().iter().map(|&v| f(v)).collect();
        missing.clone().with_labels(&y).unwrap()
    };
    let binary = labelled(|v| f32::from(v >= 1.5));
    let classes = labelled(f32::floor);
    let numeric = dataset_with(40_000, 10, true, false);
    // Weights, and labels of exactly 1 for `scale_pos_weight` to reweight.
    let weights: Vec<f32> = (0..40_000).map(|i| 0.5 + (i % 7) as f32 * 0.25).collect();
    let ones: Vec<f32> = missing
        .labels()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 3 == 0 { 1.0 } else { v })
        .collect();
    let weighted = missing
        .clone()
        .with_labels(&ones)
        .unwrap()
        .with_weights(&weights)
        .unwrap();
    // Logistic rows: the weighted set's, binarized; an odd row count, whose
    // last rows run the host's scalar path; and margins beyond the host
    // vector kernel's range on some rows (those rounds grow on the host).
    let binary_labels: Vec<f32> = ones.iter().map(|&v| f32::from(v >= 1.0)).collect();
    let weighted_binary = weighted.clone().with_labels(&binary_labels).unwrap();
    let odd = dataset(40_003, 10, true);
    let odd_labels: Vec<f32> = odd
        .labels()
        .unwrap()
        .iter()
        .map(|&v| f32::from(v >= 1.5))
        .collect();
    let odd = odd.with_labels(&odd_labels).unwrap();
    let far: Vec<f32> = (0..40_000)
        .map(|i| if i % 997 == 0 { 85.0 } else { 0.0 })
        .collect();
    let far_margins = binary.clone().with_base_margin(&far).unwrap();
    let base = || {
        TrainingParams::builder()
            .objective(Objective::SquaredError(RegLoss::default()))
            .tree_method(TreeMethod::Hist)
            .max_depth(6)
            .eta(0.3)
    };
    let monotone = vec![Monotone::None, Monotone::Increasing, Monotone::Decreasing];
    let configs: Vec<(&str, TrainingParamsBuilder, &DMatrix)> = vec![
        ("squared error", base(), &dense),
        (
            "weighted, scale_pos_weight",
            base().objective(Objective::SquaredError(RegLoss::new(2.5).unwrap())),
            &weighted,
        ),
        ("missing values", base(), &missing),
        (
            "logistic",
            base().objective(Objective::BinaryLogistic(RegLoss::default())),
            &binary,
        ),
        (
            "logistic, weighted, scale_pos_weight",
            base().objective(Objective::BinaryLogistic(RegLoss::new(2.5).unwrap())),
            &weighted_binary,
        ),
        (
            "logistic, scalar tail",
            base().objective(Objective::RegLogistic(RegLoss::default())),
            &odd,
        ),
        (
            "logitraw, margins past the vector range",
            base().objective(Objective::BinaryLogitRaw(RegLoss::default())),
            &far_margins,
        ),
        (
            "multiclass",
            base().objective(Objective::Softprob(Multiclass::new(3).unwrap())),
            &classes,
        ),
        ("subsample", base().subsample(0.7), &missing),
        (
            "gradient-based sampling",
            base()
                .subsample(0.5)
                .sampling_method(SamplingMethod::GradientBased),
            &missing,
        ),
        (
            "column sampling",
            base()
                .colsample_bytree(0.8)
                .colsample_bylevel(0.8)
                .colsample_bynode(0.8),
            &missing,
        ),
        (
            "alpha, gamma, max_delta_step",
            base()
                .alpha(0.5)
                .gamma(0.1)
                .max_delta_step(MaxDeltaStep::Bounded(0.5)),
            &missing,
        ),
        ("monotone", base().monotone_constraints(monotone), &dense),
        (
            "interaction",
            base().interaction_constraints(vec![vec![1, 2, 3], vec![4, 5]]),
            &dense,
        ),
        (
            "lossguide",
            base().grow_policy(GrowPolicy::LossGuide).max_leaves(31),
            &missing,
        ),
        (
            "symmetric",
            base().grow_policy(GrowPolicy::Symmetric),
            &numeric,
        ),
        (
            "dart",
            base().booster(BoosterKind::Dart(Dart::default())),
            &missing,
        ),
        (
            "posterior sampling",
            base().posterior_sampling(true),
            &missing,
        ),
        (
            "linear leaves",
            base().linear_tree(LinearTree::default()),
            &dense,
        ),
    ];
    for (name, builder, data) in configs {
        let cpu = builder.clone().build().unwrap();
        let gpu = builder.device(CUDA).build().unwrap();
        let train_one = |params: &TrainingParams| {
            common::with_threads(1, || train(params, data, 8).unwrap())
                .encode(ModelFormat::Binary)
                .unwrap()
        };
        assert_eq!(train_one(&cpu), train_one(&gpu), "{name}");
        assert!(
            cuda::available(),
            "{name}: {:?}",
            cuda::unavailable_reason()
        );
    }
}

/// On numeric data, depthwise trees grow resident: histograms stay on the
/// device, which subtracts siblings and scans every feature's splits. The
/// models still equal single-threaded CPU training bit for bit, across the
/// scorer's options (monotone bounds, `alpha`, `max_delta_step`,
/// `min_child_weight`), feature restrictions (interaction constraints,
/// column sampling), missing values, device-side rounds, and gradients so
/// large that scans score NaN (those nodes are searched on the host).
#[test]
fn device_cuda_resident_search_matches_single_threaded_cpu() {
    if !device() {
        return;
    }
    let dense = dataset_with(40_000, 10, false, false);
    let missing = dataset_with(40_000, 10, true, false);
    let binary_labels: Vec<f32> = missing
        .labels()
        .unwrap()
        .iter()
        .map(|&v| f32::from(v >= 1.5))
        .collect();
    let binary = missing.clone().with_labels(&binary_labels).unwrap();
    let weights: Vec<f32> = (0..40_000).map(|i| 0.5 + (i % 7) as f32 * 0.25).collect();
    let weighted = missing.clone().with_weights(&weights).unwrap();
    let huge_labels: Vec<f32> = dense.labels().unwrap().iter().map(|&v| v * 1e30).collect();
    let huge = dense.clone().with_labels(&huge_labels).unwrap();
    let base = || {
        TrainingParams::builder()
            .objective(Objective::SquaredError(RegLoss::default()))
            .tree_method(TreeMethod::Hist)
            .max_depth(6)
            .eta(0.3)
    };
    let monotone = vec![Monotone::Increasing, Monotone::None, Monotone::Decreasing];
    let configs: Vec<(&str, TrainingParamsBuilder, &DMatrix)> = vec![
        ("dense", base(), &dense),
        ("missing values", base(), &missing),
        (
            "logistic",
            base().objective(Objective::BinaryLogistic(RegLoss::default())),
            &binary,
        ),
        ("weighted", base(), &weighted),
        ("monotone", base().monotone_constraints(monotone), &missing),
        (
            "alpha, lambda, max_delta_step, min_child_weight",
            base()
                .alpha(0.5)
                .lambda(2.0)
                .max_delta_step(MaxDeltaStep::Bounded(0.5))
                .min_child_weight(3.0),
            &missing,
        ),
        (
            "interaction",
            base().interaction_constraints(vec![vec![0, 1, 2], vec![3, 4]]),
            &missing,
        ),
        (
            "column sampling",
            base()
                .colsample_bytree(0.8)
                .colsample_bylevel(0.8)
                .colsample_bynode(0.8),
            &missing,
        ),
        ("depth 1", base().max_depth(1), &missing),
        ("NaN scans", base(), &huge),
    ];
    for (name, builder, data) in configs {
        let cpu = builder.clone().build().unwrap();
        let gpu = builder.device(CUDA).build().unwrap();
        let train_one = |params: &TrainingParams| {
            common::with_threads(1, || train(params, data, 8).unwrap())
                .encode(ModelFormat::Binary)
                .unwrap()
        };
        assert_eq!(train_one(&cpu), train_one(&gpu), "{name}");
        assert!(
            cuda::available(),
            "{name}: {:?}",
            cuda::unavailable_reason()
        );
    }
}

/// At 300,000 rows every large node is summed in many chunks and the
/// partition spans many tiles per node: training there (dense and with
/// missing values, exact and inexact gradients, subsampled, depthwise and
/// loss-guided) still reproduces the CPU model bit for bit.
#[test]
fn device_cuda_training_matches_cpu_at_scale() {
    if !device() {
        return;
    }
    let dense = dataset_with(300_000, 8, false, false);
    let missing = dataset(300_000, 8, true);
    let y: Vec<f32> = missing
        .labels()
        .unwrap()
        .iter()
        .map(|&v| f32::from(v >= 1.5))
        .collect();
    let binary = missing.clone().with_labels(&y).unwrap();
    let base = || {
        TrainingParams::builder()
            .objective(Objective::SquaredError(RegLoss::default()))
            .tree_method(TreeMethod::Hist)
            .max_depth(7)
            .eta(0.3)
    };
    let logistic = || base().objective(Objective::BinaryLogistic(RegLoss::default()));
    let configs: Vec<(&str, TrainingParamsBuilder, &DMatrix)> = vec![
        ("dense", base(), &dense),
        ("dense subsample", base().subsample(0.6), &dense),
        ("missing", base(), &missing),
        ("logistic", logistic(), &binary),
        ("logistic subsample", logistic().subsample(0.5), &binary),
        (
            "lossguide",
            logistic().grow_policy(GrowPolicy::LossGuide).max_leaves(48),
            &binary,
        ),
    ];
    for (name, builder, data) in configs {
        let cpu = builder.clone().build().unwrap();
        let gpu = builder.device(CUDA).build().unwrap();
        let bytes = |params: &TrainingParams| {
            train(params, data, 4)
                .unwrap()
                .encode(ModelFormat::Binary)
                .unwrap()
        };
        assert_eq!(bytes(&cpu), bytes(&gpu), "{name}");
        assert!(
            cuda::available(),
            "{name}: {:?}",
            cuda::unavailable_reason()
        );
    }
}

/// A `device = cuda` run repeats itself exactly, independent of the worker
/// count.
#[test]
fn device_cuda_training_is_deterministic() {
    if !device() {
        return;
    }
    let data = dataset(20_000, 9, true);
    let params = TrainingParams::builder()
        .objective(Objective::SquaredError(RegLoss::default()))
        .tree_method(TreeMethod::Hist)
        .max_depth(6)
        .eta(0.3)
        .subsample(0.8)
        .device(CUDA)
        .build()
        .unwrap();
    let run = |threads| {
        common::with_threads(threads, || {
            train(&params, &data, 8)
                .unwrap()
                .encode(ModelFormat::Binary)
                .unwrap()
        })
    };
    assert_eq!(run(1), run(1));
    assert_eq!(run(1), run(4));
    assert!(cuda::available(), "{:?}", cuda::unavailable_reason());
}

/// The warp scan must preserve the first tied candidate, missing-value
/// direction, and partial windows for features wider than one warp.
#[test]
fn resident_scan_ragged_bins_and_duplicate_feature_ties_match_cpu() {
    if !device() {
        return;
    }
    let n = 20_003;
    for (max_bin, cardinality, missing) in [(17, 13, false), (33, 67, true), (257, 521, true)] {
        let mut values = Vec::with_capacity(n * 3);
        let mut labels = Vec::with_capacity(n);
        for row in 0..n {
            let x = ((row * 31) % cardinality) as f32;
            let x = if missing && row % 11 == 0 {
                f32::NAN
            } else {
                x
            };
            // Identical features make cross-feature ties exact, while the
            // first feature's bins have unequal row counts and short tails.
            values.extend([x, x, (row % 7) as f32]);
            labels.push(if x.is_nan() { -3.0 } else { (x / 8.0).floor() });
        }
        let data = DMatrix::from_dense(&values, n, 3)
            .unwrap()
            .with_labels(&labels)
            .unwrap();
        let base = TrainingParams::builder()
            .tree_method(TreeMethod::Hist)
            .max_bin(max_bin)
            .max_depth(5)
            .eta(0.2);
        let cpu = base.clone().build().unwrap();
        let gpu = base.device(CUDA).build().unwrap();
        let fit = |params: &TrainingParams| {
            common::with_threads(1, || train(params, &data, 5).unwrap())
                .encode(ModelFormat::Binary)
                .unwrap()
        };
        assert_eq!(
            fit(&cpu),
            fit(&gpu),
            "bins={max_bin}, cardinality={cardinality}, missing={missing}"
        );
        assert!(cuda::available(), "{:?}", cuda::unavailable_reason());
    }
}

fn history_bits(history: &EvalHistory) -> Vec<(usize, Vec<u64>)> {
    history
        .rounds()
        .map(|round| {
            (
                round.iteration(),
                round.values().iter().map(|value| value.to_bits()).collect(),
            )
        })
        .collect()
}

fn assert_training_bits(cpu: &TrainResult, gpu: &TrainResult) {
    assert_eq!(
        cpu.model.encode(ModelFormat::Binary).unwrap(),
        gpu.model.encode(ModelFormat::Binary).unwrap()
    );
    assert_eq!(cpu.model.best_iteration(), gpu.model.best_iteration());
    assert_eq!(
        cpu.best_score.map(f64::to_bits),
        gpu.best_score.map(f64::to_bits)
    );
    assert_eq!(cpu.history.datasets(), gpu.history.datasets());
    assert_eq!(cpu.history.metrics(), gpu.history.metrics());
    assert_eq!(cpu.history.first_iteration(), gpu.history.first_iteration());
    assert_eq!(history_bits(&cpu.history), history_bits(&gpu.history));
    for dataset in cpu.history.datasets() {
        for metric in cpu.history.metrics() {
            let bits = |history: &EvalHistory| {
                history
                    .series(dataset, metric)
                    .unwrap()
                    .map(f64::to_bits)
                    .collect::<Vec<_>>()
            };
            assert_eq!(bits(&cpu.history), bits(&gpu.history), "{dataset}/{metric}");
        }
    }
    assert!(cuda::available(), "{:?}", cuda::unavailable_reason());
}

/// Resident rounds must synchronize the margins for every eval metric and
/// hook, including a callback break, patience exhaustion, and continuation.
#[test]
fn resident_training_eval_stop_and_continuation_match_cpu() {
    if !device() {
        return;
    }
    let n = 4_096;
    let x: Vec<f32> = (0..n).map(|row| (row % 2) as f32).collect();
    let labels: Vec<f32> = x.iter().map(|&value| 4.0 * value - 2.0).collect();
    let margins: Vec<f32> = (0..n)
        .map(|row| (row % 7) as f32 * 0.0625 - 0.1875)
        .collect();
    let data = DMatrix::from_dense(&x, n, 1)
        .unwrap()
        .with_labels(&labels)
        .unwrap()
        .with_base_margin(&margins)
        .unwrap();
    // The validation target is deliberately opposed to the training target:
    // each fitted round worsens it, so patience really expires at round 2.
    let valid_labels: Vec<f32> = labels.iter().map(|&label| -label).collect();
    let valid = data.clone().with_labels(&valid_labels).unwrap();
    let base = TrainingParams::builder()
        .objective(Objective::SquaredError(RegLoss::default()))
        .tree_method(TreeMethod::Hist)
        .nthread(1)
        .max_depth(2)
        .lambda(0.0)
        .eta(0.5)
        .eval_metric(EvalMetric::Mae)
        .eval_metric(EvalMetric::Rmse);
    let cpu = base.clone().build().unwrap();
    let gpu = base.device(CUDA).build().unwrap();
    let fit = |params: &TrainingParams,
               rounds: usize,
               initial: Option<&BoostedModel>,
               patience: Option<NonZeroUsize>,
               break_at: Option<usize>| {
        let mut seen = Vec::new();
        let mut trainer = Trainer::new(params, &data, rounds)
            .eval(&data, "train")
            .eval(&valid, "valid");
        if let Some(initial) = initial {
            trainer = trainer.init_model(initial);
        }
        if let Some(patience) = patience {
            trainer = trainer.early_stopping_rounds(patience);
        }
        let out = trainer
            .on_round(|round| {
                seen.push((
                    round.iteration(),
                    round.values().iter().map(|value| value.to_bits()).collect(),
                ));
                if break_at == Some(round.iteration()) {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .train()
            .unwrap();
        assert_eq!(seen, history_bits(&out.history));
        let first = initial.map_or(0, BoostedModel::num_boost_rounds);
        assert_eq!(out.history.first_iteration(), first);
        assert_eq!(
            seen.iter()
                .map(|(iteration, _)| *iteration)
                .collect::<Vec<_>>(),
            (first..out.model.num_boost_rounds()).collect::<Vec<_>>()
        );
        out
    };
    for break_at in [Some(1), None] {
        let patience = NonZeroUsize::new(2);
        let expected = fit(&cpu, 100, None, patience, break_at);
        let actual = fit(&gpu, 100, None, patience, break_at);
        assert_training_bits(&expected, &actual);
        let rounds = if break_at.is_some() { 2 } else { 3 };
        assert_eq!(actual.model.num_boost_rounds(), rounds);
        assert_eq!(actual.history.len(), rounds);
        assert_eq!(actual.model.best_iteration(), Some(0));
        assert_eq!(
            actual.best_score.map(f64::to_bits),
            actual
                .history
                .round(0)
                .unwrap()
                .score("valid", "rmse")
                .map(f64::to_bits)
        );
        let scores: Vec<_> = actual.history.series("valid", "rmse").unwrap().collect();
        assert!(scores.windows(2).all(|pair| pair[1] > pair[0]));
    }

    let whole = fit(&cpu, 8, None, None, None);
    let whole_gpu = fit(&gpu, 8, None, None, None);
    assert_training_bits(&whole, &whole_gpu);
    let first = fit(&cpu, 100, None, None, Some(2));
    let first_gpu = fit(&gpu, 100, None, None, Some(2));
    assert_training_bits(&first, &first_gpu);
    assert_eq!(first.model.num_boost_rounds(), 3);
    let suffix: Vec<_> = history_bits(&whole.history).into_iter().skip(3).collect();
    // Both backends can resume either backend's model, with absolute hook
    // iterations and exactly the uninterrupted model and history suffix.
    for initial in [&first.model, &first_gpu.model] {
        let continued = fit(&cpu, 5, Some(initial), None, None);
        let continued_gpu = fit(&gpu, 5, Some(initial), None, None);
        assert_training_bits(&continued, &continued_gpu);
        assert_eq!(
            continued_gpu.model.encode(ModelFormat::Binary).unwrap(),
            whole.model.encode(ModelFormat::Binary).unwrap()
        );
        assert_eq!(history_bits(&continued_gpu.history), suffix);
        assert_eq!(initial.num_boost_rounds(), 3);
    }
}

/// Actual logistic inputs move across the vector kernel's +/-80 boundary,
/// rather than staying exceptional for the entire run. Prefix predictions
/// prove which round's inputs require the host and which permit residency.
#[test]
fn logistic_rounds_cross_host_resident_boundary_in_both_directions() {
    if !device() {
        return;
    }
    let n = 256;
    let rounds = 6;
    let x: Vec<f32> = (0..n).map(|row| (row / (n / 2)) as f32).collect();
    let labels: Vec<f32> = (0..n)
        .map(|row| {
            // A minority opposite label in each leaf keeps its gradient
            // nonzero after saturation, driving the next boundary crossing.
            f32::from((row < n / 2) != (row % (n / 2) == 0))
        })
        .collect();
    for (name, magnitude, delta, exceptional) in [
        (
            "host to resident",
            81.0,
            16.0,
            [true, false, false, false, false, false, false],
        ),
        (
            "resident to host and back",
            79.0,
            160.0,
            [false, true, false, true, false, true, false],
        ),
    ] {
        let margins: Vec<f32> = (0..n)
            .map(|row| {
                let sign = if row < n / 2 { -1.0 } else { 1.0 };
                sign * (magnitude + (row % 4) as f32 * 0.125 - 0.1875)
            })
            .collect();
        let data = DMatrix::from_dense(&x, n, 1)
            .unwrap()
            .with_labels(&labels)
            .unwrap()
            .with_base_margin(&margins)
            .unwrap();
        let base = TrainingParams::builder()
            .objective(Objective::BinaryLogistic(RegLoss::default()))
            .tree_method(TreeMethod::Hist)
            .nthread(1)
            .max_depth(1)
            .min_child_weight(0.0)
            .lambda(0.0)
            .eta(1.0)
            .max_delta_step(MaxDeltaStep::Bounded(delta));
        let cpu = train(&base.clone().build().unwrap(), &data, rounds).unwrap();
        let gpu = train(&base.device(CUDA).build().unwrap(), &data, rounds).unwrap();
        assert_eq!(
            cpu.encode(ModelFormat::Binary).unwrap(),
            gpu.encode(ModelFormat::Binary).unwrap(),
            "{name}"
        );
        assert_eq!(cpu.num_boost_rounds(), rounds);
        for (iteration, &outside) in exceptional.iter().enumerate() {
            let before = cpu.predict_margin(&data, ..iteration).unwrap();
            let actual = gpu.predict_margin(&data, ..iteration).unwrap();
            assert!(
                before
                    .as_slice()
                    .iter()
                    .all(|margin| { margin.is_finite() && (margin.abs() > 80.0) == outside }),
                "{name}: inputs to round {iteration} must be exceptional={outside}"
            );
            assert_eq!(
                before
                    .as_slice()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                actual
                    .as_slice()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{name}: prefix {iteration}"
            );
            if iteration == 0 {
                assert_eq!(before.as_slice(), margins.as_slice());
            }
        }
        assert!(
            cuda::available(),
            "{name}: {:?}",
            cuda::unavailable_reason()
        );
    }
}

/// Distinct resident training runs and an uploaded prediction model share
/// the CUDA device, but not thread bindings, margin buffers, or staging.
#[test]
fn concurrent_resident_training_and_prediction_match_cpu() {
    if !device() {
        return;
    }
    let a = dataset_with(8_192, 3, true, false);
    let a_margins: Vec<f32> = (0..a.n_rows())
        .map(|row| (row % 7) as f32 * 0.125)
        .collect();
    let a = a.with_base_margin(&a_margins).unwrap();
    let b = dataset_with(12_288, 4, false, false);
    let b_labels: Vec<f32> = b
        .labels()
        .unwrap()
        .iter()
        .map(|&label| f32::from(label >= 1.5))
        .collect();
    let b_margins: Vec<f32> = (0..b.n_rows())
        .map(|row| (row % 5) as f32 * 0.125 - 0.25)
        .collect();
    let b = b
        .with_labels(&b_labels)
        .unwrap()
        .with_base_margin(&b_margins)
        .unwrap();
    let base = TrainingParams::builder()
        .tree_method(TreeMethod::Hist)
        .nthread(1)
        .max_depth(3)
        .eta(0.3);
    let a_cpu = base
        .clone()
        .objective(Objective::SquaredError(RegLoss::default()))
        .build()
        .unwrap();
    let b_cpu = base
        .objective(Objective::BinaryLogistic(RegLoss::default()))
        .build()
        .unwrap();
    let mut a_gpu = a_cpu.clone();
    a_gpu.device = CUDA;
    let mut b_gpu = b_cpu.clone();
    b_gpu.device = CUDA;
    let expected_a = train(&a_cpu, &a, 6)
        .unwrap()
        .encode(ModelFormat::Binary)
        .unwrap();
    let expected_b = train(&b_cpu, &b, 6)
        .unwrap()
        .encode(ModelFormat::Binary)
        .unwrap();
    let prediction_cpu = train(&b_cpu, &b, 4).unwrap();
    let prediction_gpu = train(&b_gpu, &b, 4).unwrap();
    assert_eq!(
        prediction_cpu.encode(ModelFormat::Binary).unwrap(),
        prediction_gpu.encode(ModelFormat::Binary).unwrap()
    );
    let resident = prediction_gpu.to_cuda(0).unwrap();
    let bits = |values: &[f32]| {
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    let expected_margins = bits(prediction_cpu.predict_margin(&b, ..).unwrap().as_slice());
    let expected_predictions = bits(prediction_cpu.predict(&b, ..).unwrap().as_slice());
    let barrier = Barrier::new(3);
    std::thread::scope(|scope| {
        let fit_a = scope.spawn(|| {
            barrier.wait();
            train(&a_gpu, &a, 6)
                .unwrap()
                .encode(ModelFormat::Binary)
                .unwrap()
        });
        let fit_b = scope.spawn(|| {
            barrier.wait();
            train(&b_gpu, &b, 6)
                .unwrap()
                .encode(ModelFormat::Binary)
                .unwrap()
        });
        let predict = scope.spawn(|| {
            barrier.wait();
            for _ in 0..8 {
                assert_eq!(
                    bits(resident.predict_margin(&b, ..).unwrap().as_slice()),
                    expected_margins
                );
                assert_eq!(
                    bits(resident.predict(&b, ..).unwrap().as_slice()),
                    expected_predictions
                );
            }
        });
        assert_eq!(fit_a.join().unwrap(), expected_a);
        assert_eq!(fit_b.join().unwrap(), expected_b);
        predict.join().unwrap();
    });
    assert!(cuda::available(), "{:?}", cuda::unavailable_reason());
}

/// Training on a device ordinal that does not exist fails with a GPU error
/// instead of falling back silently.
#[test]
fn missing_device_ordinal_is_an_error() {
    if !device() {
        return;
    }
    let params = TrainingParams::builder()
        .tree_method(TreeMethod::Hist)
        .device(Device::Cuda { ordinal: 4096 })
        .build()
        .unwrap();
    let data = dataset(1_000, 3, false);
    assert!(matches!(
        train(&params, &data, 1),
        Err(hessboost::error::HessboostError::Gpu(_))
    ));
}
