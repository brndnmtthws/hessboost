//! wgpu backend integration tests (the `wgpu` feature).
//!
//! Tests that need an adapter skip when none is usable (printing why);
//! parameter-refusal tests always run. CI installs Mesa's lavapipe on its
//! Linux runners, so the GPU paths execute there without a GPU.

#![cfg(feature = "wgpu")]

mod common;

use hessboost::backend::wgpu;
use hessboost::config::{Device, MultiStrategy};
use hessboost::internals::{GHistIndex, HistogramBackend};
use hessboost::model::Predictions;
use hessboost::prelude::*;

use common::gpu::{self, GpuBackend, GpuPredictor, available};

/// The wgpu backend under the shared GPU suite.
struct Wgpu;

impl GpuBackend for Wgpu {
    const NAME: &'static str = "wgpu";
    const DEVICE: Device = Device::Wgpu;
    const MULTI_BLOCK_ROWS: usize = 600_000;
    type Model = wgpu::GpuModel;

    fn unavailable_reason() -> Option<String> {
        wgpu::unavailable_reason()
    }

    fn hist_backend(index: &GHistIndex) -> Box<dyn HistogramBackend> {
        Box::new(wgpu::WgpuHistBackend::new(index).unwrap())
    }

    fn to_gpu(model: &BoostedModel) -> Result<wgpu::GpuModel> {
        model.to_wgpu()
    }
}

impl GpuPredictor for wgpu::GpuModel {
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

/// The backend must either be fully available or absent because the machine
/// has no adapter at all. A kernel-compile, pipeline, or probe failure is
/// never an acceptable "skip" reason: without this guard, every
/// device-dependent test would pass vacuously while the backend is broken.
/// `HESSBOOST_REQUIRE_WGPU=1` (CI's Linux runners, which install lavapipe)
/// turns a missing adapter into a failure too.
#[test]
fn backend_available_or_no_adapter() {
    match wgpu::unavailable_reason() {
        None => assert!(
            wgpu::prediction_available(),
            "the wgpu adapter {:?} failed the addition-order probe, so it cannot predict",
            wgpu::device_name()
        ),
        Some(reason) => {
            assert!(
                std::env::var_os("HESSBOOST_REQUIRE_WGPU").is_none(),
                "HESSBOOST_REQUIRE_WGPU is set but the wgpu backend is unavailable: {reason}"
            );
            assert!(
                reason.starts_with("no wgpu adapter found"),
                "the wgpu backend failed to initialize: {reason}"
            );
            assert!(!wgpu::prediction_available());
        }
    }
}

/// [`gpu::training_matches_single_threaded_cpu`] on wgpu.
#[test]
fn device_wgpu_training_matches_single_threaded_cpu() {
    gpu::training_matches_single_threaded_cpu::<Wgpu>();
}

/// [`gpu::training_is_deterministic`] on wgpu.
#[test]
fn device_wgpu_training_is_deterministic() {
    gpu::training_is_deterministic::<Wgpu>();
}

/// [`gpu::refuses_unsupported_combinations`] on wgpu.
#[test]
fn device_wgpu_refuses_unsupported_combinations() {
    gpu::refuses_unsupported_combinations::<Wgpu>();
}

/// [`gpu::round_trips_through_xgboost_params`] on wgpu.
#[test]
fn device_wgpu_round_trips_through_xgboost_params() {
    gpu::round_trips_through_xgboost_params::<Wgpu>();
}

/// [`gpu::predicts_bit_identically`] on wgpu.
#[test]
fn to_wgpu_predicts_bit_identically() {
    gpu::predicts_bit_identically::<Wgpu>();
}

/// [`gpu::predicts_bit_identically_across_blocks`] on wgpu.
#[test]
fn to_wgpu_predicts_bit_identically_across_blocks() {
    gpu::predicts_bit_identically_across_blocks::<Wgpu>();
}

/// Subnormal arithmetic through the public API: a model whose leaves sit
/// around the smallest normal `f32`, predicted with and without subnormal
/// base margins, through scalar leaves (one output) and vector leaves
/// (`multi_output_tree`, several outputs). An adapter that flushes
/// subnormals in its own float adds must still give the CPU's bits.
#[test]
fn to_wgpu_predicts_subnormal_margins_bit_identically() {
    if !available::<Wgpu>() {
        return;
    }
    let n = 3_000;
    let cols = 4;
    let x: Vec<f32> = (0..n * cols)
        .map(|i: usize| ((i.wrapping_mul(2_654_435_761)) % 1000) as f32 * 0.001)
        .collect();
    let min_normal = f32::from_bits(0x0080_0000);
    let labels = |targets: usize| -> Vec<f32> {
        (0..n * targets)
            .map(|i| min_normal * (((i * 7919) % 1000) as f32 * 0.003 - 1.0))
            .collect()
    };
    let base_margin = |targets: usize| -> Vec<f32> {
        (0..n * targets)
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                f32::from_bits(((i * 48_271) % 0x00FF_FFFF) as u32) * sign
            })
            .collect()
    };
    let scalar = DMatrix::from_dense(&x, n, cols)
        .unwrap()
        .with_labels(&labels(1))
        .unwrap();
    let vector = DMatrix::from_dense(&x, n, cols)
        .unwrap()
        .with_label_matrix(&labels(3), 3)
        .unwrap();
    let params = |strategy| {
        TrainingParams::builder()
            .objective(Objective::SquaredError(RegLoss::default()))
            .tree_method(TreeMethod::Hist)
            .multi_strategy(strategy)
            .base_score(0.0)
            .max_depth(4)
            .eta(1.0)
            .build()
            .unwrap()
    };
    for (data, strategy, targets) in [
        (scalar, MultiStrategy::OneOutputPerTree, 1),
        (vector, MultiStrategy::MultiOutputTree, 3),
    ] {
        let model = train(&params(strategy), &data, 6).unwrap();
        let subnormal_leaves = model
            .trees()
            .iter()
            .flat_map(|t| {
                (0..t.nodes().len())
                    .filter(|&i| t.node(i).is_leaf())
                    .flat_map(|i| t.leaf_vector(i).iter().copied())
            })
            .filter(|v| *v != 0.0 && v.abs() < min_normal)
            .count();
        assert!(
            subnormal_leaves > 0,
            "{strategy:?}: the test needs subnormal leaves"
        );
        let gpu = model.to_wgpu().unwrap();
        let with_margin = data
            .clone()
            .with_base_margin(&base_margin(targets))
            .unwrap();
        for d in [&data, &with_margin] {
            assert_eq!(
                model.predict_margin(d, Iterations::Best).unwrap(),
                gpu.predict_margin(d, Iterations::Best).unwrap(),
                "{strategy:?}"
            );
        }
    }
}

/// [`gpu::refuses_unsupported_models`] on wgpu.
#[test]
fn to_wgpu_refuses_unsupported_models() {
    gpu::refuses_unsupported_models::<Wgpu>();
}

/// [`gpu::wide_dynamic_range_histogram_matches_cpu`] on wgpu.
#[test]
fn wide_dynamic_range_histogram_matches_cpu() {
    gpu::wide_dynamic_range_histogram_matches_cpu::<Wgpu>();
}

/// [`gpu::wide_dynamic_range_training_matches_single_threaded_cpu`] on wgpu.
#[test]
fn wide_dynamic_range_training_matches_single_threaded_cpu() {
    gpu::wide_dynamic_range_training_matches_single_threaded_cpu::<Wgpu>();
}

/// [`gpu::mismatched_inputs_match_the_cpu_backend`] on wgpu.
#[test]
fn mismatched_inputs_match_the_cpu_backend() {
    gpu::mismatched_inputs_match_the_cpu_backend::<Wgpu>();
}
