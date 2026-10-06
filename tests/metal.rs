//! Metal backend integration tests (macOS, `metal` feature).
//!
//! Tests that need a Metal device skip when one is absent — hosted macOS CI
//! runners have no GPU; parameter-refusal tests always run.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use hessboost::backend::metal;
use hessboost::config::Device;
use hessboost::internals::{GHistIndex, HistogramBackend};
use hessboost::model::Predictions;
use hessboost::prelude::*;

use common::gpu::{self, GpuBackend, GpuPredictor};

/// The Metal backend under the shared GPU suite.
struct Metal;

impl GpuBackend for Metal {
    const NAME: &'static str = "metal";
    const DEVICE: Device = Device::Metal;
    /// Enough blocks to reuse a row slot (four, at 262,144 rows each) with a
    /// last block that is short, and a handful of features so the batch stays
    /// manageable. The regression model covers the single-output arena (the
    /// 8-byte one) and the multiclass model the multi-output one (16 bytes).
    const MULTI_BLOCK_ROWS: usize = 1_200_000;
    type Model = metal::GpuModel;

    fn unavailable_reason() -> Option<String> {
        metal::unavailable_reason()
    }

    fn hist_backend(index: &GHistIndex) -> Box<dyn HistogramBackend> {
        Box::new(metal::MetalHistBackend::new(index).unwrap())
    }

    fn to_gpu(model: &BoostedModel) -> Result<metal::GpuModel> {
        model.to_gpu()
    }
}

// The across-blocks test needs four prediction blocks of 262,144 rows.
const _: () = assert!(
    Metal::MULTI_BLOCK_ROWS > 3 * 262_144,
    "the test needs four prediction blocks"
);

impl GpuPredictor for metal::GpuModel {
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
/// has no Metal device (hosted CI runners). A kernel-compile or pipeline
/// failure is never an acceptable "skip" reason: without this guard, every
/// device-dependent test would pass vacuously while the backend is broken.
#[test]
fn backend_available_or_no_device() {
    match metal::unavailable_reason() {
        None => {}
        Some(reason) => assert!(
            reason.contains("no system default Metal device"),
            "the Metal backend failed to initialize: {reason}"
        ),
    }
}

/// [`gpu::training_matches_single_threaded_cpu`] on Metal.
#[test]
fn device_metal_training_matches_single_threaded_cpu() {
    gpu::training_matches_single_threaded_cpu::<Metal>();
}

/// [`gpu::training_is_deterministic`] on Metal.
#[test]
fn device_metal_training_is_deterministic() {
    gpu::training_is_deterministic::<Metal>();
}

/// [`gpu::refuses_unsupported_combinations`] on Metal.
#[test]
fn device_metal_refuses_unsupported_combinations() {
    gpu::refuses_unsupported_combinations::<Metal>();
}

/// [`gpu::round_trips_through_xgboost_params`] on Metal.
#[test]
fn device_metal_round_trips_through_xgboost_params() {
    gpu::round_trips_through_xgboost_params::<Metal>();
}

/// [`gpu::predicts_bit_identically`] on Metal.
#[test]
fn to_gpu_predicts_bit_identically() {
    gpu::predicts_bit_identically::<Metal>();
}

/// [`gpu::predicts_bit_identically_across_blocks`] on Metal.
#[test]
fn to_gpu_predicts_bit_identically_across_blocks() {
    gpu::predicts_bit_identically_across_blocks::<Metal>();
}

/// [`gpu::refuses_unsupported_models`] on Metal.
#[test]
fn to_gpu_refuses_unsupported_models() {
    gpu::refuses_unsupported_models::<Metal>();
}

/// [`gpu::wide_dynamic_range_histogram_matches_cpu`] on Metal.
#[test]
fn wide_dynamic_range_histogram_matches_cpu() {
    gpu::wide_dynamic_range_histogram_matches_cpu::<Metal>();
}

/// [`gpu::wide_dynamic_range_training_matches_single_threaded_cpu`] on Metal.
#[test]
fn wide_dynamic_range_training_matches_single_threaded_cpu() {
    gpu::wide_dynamic_range_training_matches_single_threaded_cpu::<Metal>();
}

/// [`gpu::mismatched_inputs_match_the_cpu_backend`] on Metal.
#[test]
fn mismatched_inputs_match_the_cpu_backend() {
    gpu::mismatched_inputs_match_the_cpu_backend::<Metal>();
}
