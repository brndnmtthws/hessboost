//! CUDA backend integration tests (Linux, `cuda` feature): the shared GPU
//! suite of `common::gpu` on device 0, then the CUDA-specific cases: every
//! histogram strategy, a matrix of training configurations, resident split
//! search, training at scale, ragged-bin ties, evaluation with early
//! stopping and continuation, the logistic host/resident boundary,
//! concurrent training and prediction, a missing device ordinal, and
//! prediction edge cases (sentinel missing values, CSR input, base margins,
//! iteration ranges, concurrent calls, subnormals, model shrinkage).
//!
//! Tests that need the device skip, printing why, only when the machine
//! lacks what the backend needs (no driver, a driver or device older than
//! the backend supports, no device 0); any other reason fails them, and so
//! does any reason when `HESSBOOST_REQUIRE_CUDA` is set. Parameter tests
//! always run.

#![cfg(all(target_os = "linux", feature = "cuda"))]

mod common;

use hessboost::backend::cuda::{self, CudaHistBackend};
use hessboost::config::{
    BoosterKind, Dart, Device, GrowPolicy, LinearTree, MaxDeltaStep, ModelShrink, ModelShrinkMode,
    Monotone, MultiStrategy, SamplingMethod, TrainingParamsBuilder,
};
use hessboost::internals::{
    CpuBackend, GHistIndex, HistCuts, HistogramBackend, NodeCounts, zeroed,
};
use hessboost::model::Predictions;
use hessboost::objective::{GradPair, Multiclass, RegLoss};
use hessboost::prelude::*;
use hessboost::training::{EvalHistory, TrainResult};
use std::num::NonZeroUsize;
use std::ops::ControlFlow;
use std::sync::Barrier;

use common::bits::bits;
use common::gpu::{self, GpuBackend, GpuPredictor};

/// The CUDA backend (device 0) under the shared GPU suite.
struct Cuda;

impl GpuBackend for Cuda {
    const NAME: &'static str = "cuda";
    const DEVICE: Device = Device::Cuda { ordinal: 0 };
    /// Five 16,384-row prediction blocks, the last one short.
    const MULTI_BLOCK_ROWS: usize = 70_001;
    type Model = cuda::GpuModel;

    /// Why training or prediction cannot run on device 0. Panics unless the
    /// reason is the machine's (so a broken backend never skips) and
    /// `HESSBOOST_REQUIRE_CUDA` is unset.
    fn unavailable_reason() -> Option<String> {
        let reason =
            cuda::unavailable_reason(0).or_else(|| cuda::prediction_unavailable_reason(0))?;
        assert!(
            reason.is_environment(),
            "the CUDA backend failed to initialize: {reason}"
        );
        assert!(
            std::env::var_os("HESSBOOST_REQUIRE_CUDA").is_none(),
            "HESSBOOST_REQUIRE_CUDA is set but CUDA is unavailable: {reason}"
        );
        Some(reason.to_string())
    }

    fn hist_backend(index: &GHistIndex) -> Box<dyn HistogramBackend> {
        Box::new(CudaHistBackend::new(index, 0).unwrap())
    }

    fn to_gpu(model: &BoostedModel) -> Result<cuda::GpuModel> {
        model.to_cuda(0)
    }
}

impl GpuPredictor for cuda::GpuModel {
    fn predict(&self, data: &DMatrix, iterations: Iterations) -> Result<Predictions> {
        self.predict(data, iterations)
    }

    fn predict_margin(&self, data: &DMatrix, iterations: Iterations) -> Result<Predictions> {
        self.predict_margin(data, iterations)
    }

    fn predict_class(&self, data: &DMatrix, iterations: Iterations) -> Result<Predictions<u32>> {
        self.predict_class(data, iterations)
    }
}

/// Training and prediction agree on whether device 0 can run, and a reason
/// it cannot is the machine's, never a backend failure: without this guard,
/// every device test would pass vacuously while the backend is broken.
/// `HESSBOOST_REQUIRE_CUDA=1` turns any reason into a failure.
#[test]
fn backend_available_or_no_device() {
    assert_eq!(
        cuda::unavailable_reason(0),
        cuda::prediction_unavailable_reason(0)
    );
    // Panics on a reason that is not the machine's.
    gpu::available::<Cuda>();
}

/// [`gpu::training_matches_single_threaded_cpu`] on CUDA.
#[test]
fn device_cuda_training_matches_single_threaded_cpu() {
    gpu::training_matches_single_threaded_cpu::<Cuda>();
}

/// [`gpu::training_is_deterministic`] on CUDA.
#[test]
fn device_cuda_training_is_deterministic() {
    gpu::training_is_deterministic::<Cuda>();
}

/// [`gpu::refuses_unsupported_combinations`] on CUDA.
#[test]
fn device_cuda_refuses_unsupported_combinations() {
    gpu::refuses_unsupported_combinations::<Cuda>();
}

/// [`gpu::round_trips_through_xgboost_params`] on CUDA.
#[test]
fn device_cuda_round_trips_through_xgboost_params() {
    gpu::round_trips_through_xgboost_params::<Cuda>();
}

/// [`gpu::predicts_bit_identically`] on CUDA.
#[test]
fn to_cuda_predicts_bit_identically() {
    gpu::predicts_bit_identically::<Cuda>();
}

/// [`gpu::predicts_bit_identically_across_blocks`] on CUDA.
#[test]
fn to_cuda_predicts_bit_identically_across_blocks() {
    gpu::predicts_bit_identically_across_blocks::<Cuda>();
}

/// [`gpu::refuses_unsupported_models`] on CUDA.
#[test]
fn to_cuda_refuses_unsupported_models() {
    gpu::refuses_unsupported_models::<Cuda>();
}

/// [`gpu::wide_dynamic_range_histogram_matches_cpu`] on CUDA.
#[test]
fn wide_dynamic_range_histogram_matches_cpu() {
    gpu::wide_dynamic_range_histogram_matches_cpu::<Cuda>();
}

/// [`gpu::wide_dynamic_range_training_matches_single_threaded_cpu`] on CUDA.
#[test]
fn wide_dynamic_range_training_matches_single_threaded_cpu() {
    gpu::wide_dynamic_range_training_matches_single_threaded_cpu::<Cuda>();
}

/// [`gpu::mismatched_inputs_match_the_cpu_backend`] on CUDA.
#[test]
fn mismatched_inputs_match_the_cpu_backend() {
    gpu::mismatched_inputs_match_the_cpu_backend::<Cuda>();
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
/// two of them is exact, so every node takes an `f64` chain path (the
/// GPU's, or the CPU's for a chain of 8,192 rows or more).
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

/// Where a histogram case's node must be summed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Built {
    /// By one of the CUDA strategies.
    Gpu,
    /// By the CPU backend.
    Cpu,
}

/// Every strategy's histograms equal the CPU backend's bit for bit, under
/// the CPU's summation order: a node below 8,192 rows is one row-order
/// chain; a dense index sums a contiguous row range, or any row subset of
/// an index of at most 2^18 rows, as row-order chains; every other node
/// sums fixed blocks. CUDA sums exact nodes as integers anywhere, blocked
/// nodes as exact chunks or chains, and chains below 8,192 rows with its
/// chain kernel; a non-exact chain of 8,192 rows or more, and a non-finite
/// gradient, are built on the CPU. The node counts show each strategy ran.
#[test]
fn histograms_match_cpu_for_every_strategy() {
    if !gpu::available::<Cuda>() {
        return;
    }
    let mut seen = NodeCounts::default();
    let mut check =
        |index: &GHistIndex, rows: &[u32], gpair: &[GradPair], built: Built, what: &str| {
            let gpu = CudaHistBackend::new(index, 0).unwrap();
            let cpu = histogram(&CpuBackend, index, rows, gpair);
            assert_eq!(histogram(&gpu, index, rows, gpair), cpu, "{what}");
            let counts = gpu.node_counts();
            match built {
                Built::Gpu => assert_eq!(counts.cpu_nodes, 0, "{what}: must run on CUDA"),
                Built::Cpu => assert!(counts.cpu_nodes > 0, "{what}: must run on the CPU"),
            }
            assert!(
                cuda::available(0),
                "{what}: {:?}",
                cuda::unavailable_reason(0)
            );
            seen.exact_nodes += counts.exact_nodes;
            seen.exact_chunk_nodes += counts.exact_chunk_nodes;
            seen.chain_nodes += counts.chain_nodes;
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

    // A dense index of at most 2^18 rows: every node is a chain.
    let small = index(60_000, 6, 256, value);
    let all: Vec<u32> = (0..60_000).collect();
    let thirds: Vec<u32> = (0..60_000).step_by(3).collect();
    let few: Vec<u32> = (0..60_000).step_by(11).take(5000).collect();
    check(
        &small,
        &all,
        &exact_pairs(60_000),
        Built::Gpu,
        "dense chain, exact",
    );
    check(
        &small,
        &few,
        &wide_pairs(60_000),
        Built::Gpu,
        "dense small node, chains",
    );
    check(
        &small,
        &thirds,
        &wide_pairs(60_000),
        Built::Cpu,
        "dense subset, long chain, cpu",
    );

    // A dense index above 2^18 rows: a row range is a chain, a subset of
    // 8,192 rows or more is blocked.
    let large = index(300_000, 4, 64, value);
    let range: Vec<u32> = (0..300_000).collect();
    let half: Vec<u32> = (0..300_000).step_by(2).collect();
    check(
        &large,
        &range,
        &wide_pairs(300_000),
        Built::Cpu,
        "dense range, long chain, cpu",
    );
    check(
        &large,
        &half,
        &exact_pairs(300_000),
        Built::Gpu,
        "dense blocked, exact",
    );
    check(
        &large,
        &half,
        &chunk_exact_pairs(300_000),
        Built::Gpu,
        "dense blocked, exact chunks",
    );
    check(
        &large,
        &half,
        &wide_pairs(300_000),
        Built::Gpu,
        "dense blocked, chains",
    );

    // Missing values (a mostly full index, and a CSR-only one): a node of
    // 8,192 rows or more is blocked, a smaller one a chain.
    for (name, cells) in [
        ("missing", &with_missing as &dyn Fn(usize, usize) -> f32),
        ("csr", &sparse),
    ] {
        let idx = index(40_000, 7, 128, cells);
        let rows: Vec<u32> = (0..40_000).filter(|r| r % 7 != 3).collect();
        check(
            &idx,
            &rows,
            &exact_pairs(40_000),
            Built::Gpu,
            &format!("{name}, exact"),
        );
        check(
            &idx,
            &rows,
            &chunk_exact_pairs(40_000),
            Built::Gpu,
            &format!("{name}, exact chunks"),
        );
        check(
            &idx,
            &rows,
            &wide_pairs(40_000),
            Built::Gpu,
            &format!("{name}, blocked chains"),
        );
        check(
            &idx,
            &rows[..3000],
            &wide_pairs(40_000),
            Built::Gpu,
            &format!("{name}, small chains"),
        );
    }

    // Many features: 300 of 256 bins each, 76,800 bins in total.
    let many = index(20_000, 300, 256, |r, f| ((r * 31 + f * 7) % 20_000) as f32);
    let rows: Vec<u32> = (0..20_000).collect();
    check(
        &many,
        &rows,
        &exact_pairs(20_000),
        Built::Gpu,
        "many features, exact",
    );
    check(
        &many,
        &rows[..4000],
        &wide_pairs(20_000),
        Built::Gpu,
        "many features, small chains",
    );

    // A non-finite gradient anywhere in the slice: the CPU's NaN bits.
    let mut nan = exact_pairs(60_000);
    nan[11].grad = f32::NAN;
    nan[22].hess = f32::INFINITY;
    check(&small, &few, &nan, Built::Cpu, "non-finite, cpu");

    assert!(seen.exact_nodes > 0, "{seen:?}");
    assert!(seen.exact_chunk_nodes > 0, "{seen:?}");
    assert!(seen.chain_nodes > 0, "{seen:?}");
    assert!(cuda::available(0), "{:?}", cuda::unavailable_reason(0));
}

/// An index with the uploaded one's dimensions and cut count but different
/// bin contents is uploaded afresh, not served from the device copy.
#[test]
fn same_shape_index_with_other_bins_is_not_reused() {
    if !gpu::available::<Cuda>() {
        return;
    }
    let n = 10_000;
    let uploaded = index(n, 1, 256, |r, _| (r % 5) as f32);
    let other = index(n, 1, 256, |r, _| ((r + 1) % 5) as f32);
    let gpair: Vec<_> = (0..n)
        .map(|i| GradPair::new((i % 7) as f32 - 3.0, 1.0))
        .collect();
    let rows: Vec<u32> = (0..n as u32).collect();
    let backend = CudaHistBackend::new(&uploaded, 0).unwrap();
    let first = histogram(&CpuBackend, &uploaded, &rows, &gpair);
    assert_eq!(histogram(&backend, &uploaded, &rows, &gpair), first);
    let expected = histogram(&CpuBackend, &other, &rows, &gpair);
    assert_ne!(expected, first);
    assert_eq!(histogram(&backend, &other, &rows, &gpair), expected);
    assert!(cuda::available(0), "{:?}", cuda::unavailable_reason(0));
}

/// `device = cuda` training reproduces single-threaded CPU training bit for
/// bit, tree for tree (the whole serialized model compares equal), across
/// the configurations the hist builder serves.
#[test]
fn device_cuda_configurations_match_single_threaded_cpu() {
    if !gpu::available::<Cuda>() {
        return;
    }
    let dense = gpu::dataset_with(40_000, 10, false, true);
    let missing = gpu::dataset(40_000, 10);
    let labelled = |f: fn(f32) -> f32| {
        let y: Vec<f32> = missing.labels().unwrap().iter().map(|&v| f(v)).collect();
        missing.clone().with_labels(&y).unwrap()
    };
    let binary = labelled(|v| f32::from(v >= 1.5));
    let classes = labelled(f32::floor);
    let numeric = gpu::dataset_with(40_000, 10, true, false);
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
    let odd = gpu::dataset(40_003, 10);
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
        let gpu = builder.device(Cuda::DEVICE).build().unwrap();
        let train_one = |params: &TrainingParams| {
            common::with_threads(1, || train(params, data, 8).unwrap())
                .encode(ModelFormat::Binary)
                .unwrap()
        };
        assert_eq!(train_one(&cpu), train_one(&gpu), "{name}");
        assert!(
            cuda::available(0),
            "{name}: {:?}",
            cuda::unavailable_reason(0)
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
    if !gpu::available::<Cuda>() {
        return;
    }
    let dense = gpu::dataset_with(40_000, 10, false, false);
    let missing = gpu::dataset_with(40_000, 10, true, false);
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
        let gpu = builder.device(Cuda::DEVICE).build().unwrap();
        let train_one = |params: &TrainingParams| {
            common::with_threads(1, || train(params, data, 8).unwrap())
                .encode(ModelFormat::Binary)
                .unwrap()
        };
        assert_eq!(train_one(&cpu), train_one(&gpu), "{name}");
        assert!(
            cuda::available(0),
            "{name}: {:?}",
            cuda::unavailable_reason(0)
        );
    }
}

/// At 300,000 rows every large node is summed in many chunks and the
/// partition spans many tiles per node: training there (dense and with
/// missing values, exact and inexact gradients, subsampled, depthwise and
/// loss-guided) still reproduces the CPU model bit for bit.
#[test]
fn device_cuda_training_matches_cpu_at_scale() {
    if !gpu::available::<Cuda>() {
        return;
    }
    let dense = gpu::dataset_with(300_000, 8, false, false);
    let missing = gpu::dataset(300_000, 8);
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
        let gpu = builder.device(Cuda::DEVICE).build().unwrap();
        let bytes = |params: &TrainingParams| {
            train(params, data, 4)
                .unwrap()
                .encode(ModelFormat::Binary)
                .unwrap()
        };
        assert_eq!(bytes(&cpu), bytes(&gpu), "{name}");
        assert!(
            cuda::available(0),
            "{name}: {:?}",
            cuda::unavailable_reason(0)
        );
    }
}

/// The warp scan must preserve the first tied candidate, missing-value
/// direction, and partial windows for features wider than one warp.
#[test]
fn resident_scan_ragged_bins_and_duplicate_feature_ties_match_cpu() {
    if !gpu::available::<Cuda>() {
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
        let gpu = base.device(Cuda::DEVICE).build().unwrap();
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
        assert!(cuda::available(0), "{:?}", cuda::unavailable_reason(0));
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
    assert!(cuda::available(0), "{:?}", cuda::unavailable_reason(0));
}

/// Resident rounds must synchronize the margins for every eval metric and
/// hook, including a callback break, patience exhaustion, and continuation.
#[test]
fn resident_training_eval_stop_and_continuation_match_cpu() {
    if !gpu::available::<Cuda>() {
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
    let gpu = base.device(Cuda::DEVICE).build().unwrap();
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
    if !gpu::available::<Cuda>() {
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
        let gpu = train(&base.device(Cuda::DEVICE).build().unwrap(), &data, rounds).unwrap();
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
            cuda::available(0),
            "{name}: {:?}",
            cuda::unavailable_reason(0)
        );
    }
}

/// Distinct resident training runs and an uploaded prediction model share
/// the CUDA device, but not thread bindings, margin buffers, or staging.
#[test]
fn concurrent_resident_training_and_prediction_match_cpu() {
    if !gpu::available::<Cuda>() {
        return;
    }
    let a = gpu::dataset_with(8_192, 3, true, false);
    let a_margins: Vec<f32> = (0..a.n_rows())
        .map(|row| (row % 7) as f32 * 0.125)
        .collect();
    let a = a.with_base_margin(&a_margins).unwrap();
    let b = gpu::dataset_with(12_288, 4, false, false);
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
    a_gpu.device = Cuda::DEVICE;
    let mut b_gpu = b_cpu.clone();
    b_gpu.device = Cuda::DEVICE;
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
    assert!(cuda::available(0), "{:?}", cuda::unavailable_reason(0));
}

/// Training on a device ordinal that does not exist fails with a GPU error
/// instead of falling back silently.
#[test]
fn missing_device_ordinal_is_an_error() {
    if !gpu::available::<Cuda>() {
        return;
    }
    let params = TrainingParams::builder()
        .tree_method(TreeMethod::Hist)
        .device(Device::Cuda { ordinal: 4096 })
        .build()
        .unwrap();
    let data = gpu::dataset(1_000, 3);
    assert!(matches!(
        train(&params, &data, 1),
        Err(hessboost::error::HessboostError::Gpu(_))
    ));
}

/// `expected` and `actual` hold the same bits.
fn assert_bits(expected: Predictions, actual: Predictions) {
    assert_eq!(bits(expected.into_vec()), bits(actual.into_vec()));
}

/// Predictions on device 0 equal the model's bit for bit for dense input
/// with a sentinel missing value and the same rows as CSR, with base
/// margins, over iteration ranges, and from concurrent calls on one
/// predictor; out-of-range iterations are refused as the CPU refuses them.
#[test]
fn dense_sentinel_sparse_base_margins_ranges_and_concurrent_calls() {
    if !gpu::available::<Cuda>() {
        return;
    }
    let rows = 40_003;
    let cols = 4;
    let mut dense = Vec::with_capacity(rows * cols);
    let mut ptr = vec![0];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    let mut labels = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut target = 0.0;
        for col in 0..cols {
            if (row + col) % 3 == 0 {
                dense.push(-999.0);
            } else {
                let value = ((row * 97 + col * 13) % 1000) as f32 / 1000.0;
                dense.push(value);
                indices.push(col as u32);
                values.push(value);
                target += value;
            }
        }
        ptr.push(values.len());
        labels.push(target);
    }
    let train_data = DMatrix::from_dense_with_missing(&dense, rows, cols, -999.0)
        .unwrap()
        .with_labels(&labels)
        .unwrap();
    let sparse = DMatrix::from_csr(ptr, indices, values, cols).unwrap();
    for objective in [
        Objective::SquaredError(RegLoss::default()),
        Objective::Softprob(Multiclass::new(3).unwrap()),
    ] {
        let multiclass = objective.num_class().is_some();
        let labels = if multiclass {
            labels.iter().map(|v| v.trunc()).collect::<Vec<_>>()
        } else {
            labels.clone()
        };
        let data = train_data.clone().with_labels(&labels).unwrap();
        let params = TrainingParams::builder()
            .objective(objective)
            .num_parallel_tree(2)
            .max_depth(3)
            .build()
            .unwrap();
        let model = train(&params, &data, 6).unwrap();
        let gpu = model.to_cuda(0).unwrap();
        let outputs = model.n_outputs();
        let base: Vec<f32> = (0..rows * outputs)
            .map(|i| (i % 17) as f32 * 0.013 - 0.1)
            .collect();
        let dense = data.with_base_margin(&base).unwrap();
        let sparse = sparse.clone().with_base_margin(&base).unwrap();
        for iterations in [
            Iterations::Best,
            Iterations::from(..),
            Iterations::from(0..0),
            Iterations::from(2..5),
        ] {
            assert_bits(
                model.predict_margin(&dense, iterations).unwrap(),
                gpu.predict_margin(&dense, iterations).unwrap(),
            );
            assert_bits(
                model.predict_margin(&sparse, iterations).unwrap(),
                gpu.predict_margin(&sparse, iterations).unwrap(),
            );
        }
        std::thread::scope(|scope| {
            let first = scope.spawn(|| gpu.predict_margin(&dense, 1..6).unwrap());
            let second = scope.spawn(|| gpu.predict_margin(&sparse, 2..5).unwrap());
            assert_bits(
                model.predict_margin(&dense, 1..6).unwrap(),
                first.join().unwrap(),
            );
            assert_bits(
                model.predict_margin(&sparse, 2..5).unwrap(),
                second.join().unwrap(),
            );
        });
        // A range past the model's 6 iterations, and one starting there
        // (an inverted range).
        assert_eq!(
            common::incompatible_model(model.predict_margin(&dense, ..7)),
            "iterations"
        );
        assert_eq!(
            common::incompatible_model(gpu.predict_margin(&dense, ..7)),
            "iterations"
        );
        assert_eq!(
            common::invalid_param(model.predict_margin(&dense, 7..)),
            "iterations"
        );
        assert_eq!(
            common::invalid_param(gpu.predict_margin(&dense, 7..)),
            "iterations"
        );
    }
}

/// Leaves around the smallest normal `f32`, with subnormal base margins,
/// predict on device 0 bit-identically to the CPU through scalar leaves
/// (one output per tree) and vector leaves (multi-output trees).
#[test]
fn scalar_and_vector_subnormal_leaves_and_margins() {
    if !gpu::available::<Cuda>() {
        return;
    }
    let rows = 3_001;
    let cols = 4;
    let x: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 7919) % 1000) as f32 * 0.001)
        .collect();
    let min_normal = f32::from_bits(0x0080_0000);
    for (targets, strategy) in [
        (1, MultiStrategy::OneOutputPerTree),
        (3, MultiStrategy::MultiOutputTree),
    ] {
        let labels: Vec<f32> = (0..rows * targets)
            .map(|i| min_normal * (((i * 97) % 1000) as f32 * 0.003 - 1.0))
            .collect();
        let data = DMatrix::from_dense(&x, rows, cols)
            .unwrap()
            .with_label_matrix(&labels, targets)
            .unwrap();
        let params = TrainingParams::builder()
            .multi_strategy(strategy)
            .base_score(0.0)
            .max_depth(4)
            .eta(1.0)
            .build()
            .unwrap();
        let model = train(&params, &data, 6).unwrap();
        let subnormal = model
            .trees()
            .iter()
            .flat_map(|tree| {
                tree.nodes()
                    .iter()
                    .enumerate()
                    .filter(|(_, node)| node.is_leaf())
                    .flat_map(|(id, _)| tree.leaf_vector(id))
            })
            .any(|v| *v != 0.0 && v.abs() < min_normal);
        assert!(subnormal, "test must exercise subnormal leaf arithmetic");
        let base: Vec<f32> = (0..rows * targets)
            .map(|i| match i % 4 {
                0 => -0.0,
                1 => f32::from_bits((i % 0x0080_0000) as u32),
                2 => -f32::from_bits((i % 0x0080_0000) as u32),
                _ => 0.0,
            })
            .collect();
        let data = data.with_base_margin(&base).unwrap();
        let gpu = model.to_cuda(0).unwrap();
        assert_bits(
            model.predict_margin(&data, ..).unwrap(),
            gpu.predict_margin(&data, ..).unwrap(),
        );
    }
}

/// A model trained with model shrinkage predicts on device 0 as on the
/// CPU, and both refuse a range that starts after iteration 0.
#[test]
fn explicit_model_shrinkage_cpu_convention() {
    if !gpu::available::<Cuda>() {
        return;
    }
    let data = gpu::dataset(300, 4);
    let params = TrainingParams::builder()
        .model_shrink(ModelShrink::new(0.1, ModelShrinkMode::Decreasing).unwrap())
        .build()
        .unwrap();
    let model = train(&params, &data, 5).unwrap();
    let gpu = model.to_cuda(0).unwrap();
    for iterations in [
        Iterations::Best,
        Iterations::from(0..0),
        Iterations::from(..3),
    ] {
        assert_bits(
            model.predict_margin(&data, iterations).unwrap(),
            gpu.predict_margin(&data, iterations).unwrap(),
        );
    }
    for refused in [
        model.predict_margin(&data, 2..5),
        gpu.predict_margin(&data, 2..5),
    ] {
        assert_eq!(common::incompatible_model(refused), "iterations");
    }
}
