//! CUDA prediction parity and multi-stream lifetime boundaries.
#![cfg(all(target_os = "linux", feature = "cuda"))]

mod common;

use common::gpu::{self, GpuBackend, GpuPredictor};
use hessboost::backend::cuda;
use hessboost::config::{Device, ModelShrink, ModelShrinkMode, MultiStrategy};
use hessboost::internals::{GHistIndex, HistogramBackend};
use hessboost::model::Predictions;
use hessboost::objective::{Multiclass, RegLoss};
use hessboost::prelude::*;

struct Cuda;

impl GpuBackend for Cuda {
    const NAME: &'static str = "cuda";
    const DEVICE: Device = Device::Cuda { ordinal: 0 };
    const MULTI_BLOCK_ROWS: usize = 70_001;
    type Model = cuda::GpuModel;

    fn unavailable_reason() -> Option<String> {
        if cuda::prediction_available(0) {
            return None;
        }
        let data = gpu::dataset(8, 2);
        let model = train(&TrainingParams::default(), &data, 1).unwrap();
        let reason = model.to_cuda(0).unwrap_err().to_string();
        assert!(
            std::env::var_os("HESSBOOST_REQUIRE_CUDA").is_none(),
            "{reason}"
        );
        assert!(
            ["libcuda not found", "libnvrtc not found", "no CUDA device"]
                .iter()
                .any(|expected| reason.contains(expected)),
            "CUDA predictor failed to initialize: {reason}"
        );
        Some(reason)
    }

    fn hist_backend(index: &GHistIndex) -> Box<dyn HistogramBackend> {
        Box::new(cuda::CudaHistBackend::new(index, 0).unwrap())
    }

    fn to_gpu(model: &BoostedModel) -> Result<Self::Model> {
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

#[test]
fn categorical_dart_multiclass_cpu_parity() {
    gpu::predicts_bit_identically::<Cuda>();
}

#[test]
fn several_pinned_batches_cpu_parity() {
    gpu::predicts_bit_identically_across_blocks::<Cuda>();
}

#[test]
fn linear_models_are_refused() {
    gpu::refuses_unsupported_models::<Cuda>();
}

fn assert_bits(expected: Predictions, actual: Predictions) {
    assert_eq!(
        expected
            .into_vec()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        actual
            .into_vec()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>()
    );
}

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
        assert!(gpu.predict_margin(&dense, 7..).is_err());
    }
}

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
        Iterations::from(2..5),
    ] {
        let expected = model.predict_margin(&data, iterations);
        let actual = gpu.predict_margin(&data, iterations);
        match (expected, actual) {
            (Ok(expected), Ok(actual)) => assert_bits(expected, actual),
            (Err(expected), Err(actual)) => assert_eq!(expected.to_string(), actual.to_string()),
            other => panic!("shrinkage paths disagree: {other:?}"),
        }
    }
}
