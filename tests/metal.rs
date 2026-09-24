//! Metal backend integration tests (macOS, `metal` feature).
//!
//! Tests that need a Metal device skip when one is absent — hosted macOS CI
//! runners have no GPU; parameter-refusal tests always run.

#![cfg(all(target_os = "macos", feature = "metal"))]

use hessboost::backend::metal;
use hessboost::config::{BoosterKind, Device, ProcessType, TreeMethod};
use hessboost::prelude::*;

/// Whether a Metal device is present, with the skip reason printed so a
/// vacuous pass is visible.
fn device() -> bool {
    if let Some(reason) = metal::unavailable_reason() {
        eprintln!("skipping metal test: {reason}");
        return false;
    }
    true
}

/// The backend must either be fully available or absent because the machine
/// has no Metal device (hosted CI runners). A kernel-compile or pipeline
/// failure is never an acceptable "skip" reason: without this guard, every
/// device-dependent test above would pass vacuously while the backend is
/// broken.
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

/// A deterministic regression dataset with missing values and a categorical
/// first column.
fn dataset(n: usize, cols: usize) -> DMatrix {
    let mut x = vec![0.0f32; n * cols];
    let mut y = vec![0.0f32; n];
    for r in 0..n {
        let mut target = 0.0;
        for f in 0..cols {
            let v = if f == 0 {
                ((r * 31 + f) % 5) as f32 // categorical codes
            } else if (r + f) % 13 == 0 {
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
            if f == 0 {
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

/// `device = metal` training reproduces single-threaded CPU training bit for
/// bit, tree for tree (the whole serialized model compares equal).
#[test]
fn device_metal_training_matches_single_threaded_cpu() {
    if !device() {
        return;
    }
    let data = dataset(40_000, 12);
    let build = |device| {
        TrainingParams::builder()
            .objective("reg:squarederror")
            .tree_method(TreeMethod::Hist)
            .max_depth(6)
            .eta(0.3)
            .device(device)
            .build()
            .unwrap()
    };
    let train_one = |params| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| train(&params, &data, 10).unwrap())
    };
    let cpu = train_one(build(Device::Cpu));
    let gpu = train_one(build(Device::Metal));
    assert_eq!(
        cpu.to_bytes().unwrap(),
        gpu.to_bytes().unwrap(),
        "the metal-trained model must be bit-identical to the CPU's"
    );
}

/// A `device = metal` run repeats itself exactly, independent of the worker
/// count.
#[test]
fn device_metal_training_is_deterministic() {
    if !device() {
        return;
    }
    let data = dataset(20_000, 9);
    let params = TrainingParams::builder()
        .objective("reg:squarederror")
        .tree_method(TreeMethod::Hist)
        .max_depth(6)
        .eta(0.3)
        .subsample(0.8)
        .device(Device::Metal)
        .build()
        .unwrap();
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| train(&params, &data, 8).unwrap().to_bytes().unwrap())
    };
    assert_eq!(run(1), run(1));
    assert_eq!(run(1), run(4));
}

/// The unsupported `device = metal` combinations are refused with an error,
/// never silently ignored.
#[test]
fn device_metal_refuses_unsupported_combinations() {
    let base = TrainingParams::builder()
        .device(Device::Metal)
        .build()
        .unwrap();
    let variants: Vec<(TrainingParams, &str)> = vec![
        (
            TrainingParams {
                tree_method: TreeMethod::Approx,
                ..base.clone()
            },
            "tree_method=approx",
        ),
        (
            TrainingParams {
                tree_method: TreeMethod::Exact,
                ..base.clone()
            },
            "tree_method=exact",
        ),
        (
            TrainingParams {
                use_quantized_grad: true,
                ..base.clone()
            },
            "use_quantized_grad",
        ),
        (
            TrainingParams {
                booster: BoosterKind::GbLinear,
                ..base.clone()
            },
            "booster=gblinear",
        ),
        (
            TrainingParams {
                process_type: ProcessType::Update,
                ..base.clone()
            },
            "process_type=update",
        ),
    ];
    for (params, name) in variants {
        let err = params.validate().unwrap_err().to_string();
        assert!(err.contains("device"), "{name}: {err}");
    }
}

/// `to_gpu` predictions are bit-identical to the model's across objectives,
/// missing values, categorical splits, DART weights, and iteration ranges.
#[test]
fn to_gpu_predicts_bit_identically() {
    if !device() {
        return;
    }
    let cases: Vec<(&str, usize)> = [
        ("reg:squarederror", 0),
        ("binary:logistic", 0),
        ("multi:softmax", 4),
    ]
    .to_vec();
    for (objective, num_class) in cases {
        let data = dataset(6_000, 7);
        let labels: Vec<f32> = data
            .labels()
            .unwrap()
            .iter()
            .map(|&y| {
                if num_class > 0 {
                    y.trunc() % num_class as f32
                } else if objective == "binary:logistic" {
                    f32::from(y >= 1.5)
                } else {
                    y
                }
            })
            .collect();
        let data = data.with_labels(&labels).unwrap();
        let mut builder = TrainingParams::builder()
            .objective(objective)
            .tree_method(TreeMethod::Hist)
            .max_depth(5)
            .eta(0.4)
            .booster(BoosterKind::Dart);
        if num_class > 0 {
            builder = builder.num_class(num_class);
        }
        let model = train(&builder.build().unwrap(), &data, 15).unwrap();
        let gpu = model.to_gpu().unwrap();
        assert_eq!(
            model.predict(&data).unwrap(),
            gpu.predict(&data).unwrap(),
            "{objective}: predict"
        );
        assert_eq!(
            model.predict_margin(&data).unwrap(),
            gpu.predict_margin(&data).unwrap(),
            "{objective}: predict_margin"
        );
        assert_eq!(
            model.predict_class(&data).unwrap(),
            gpu.predict_class(&data).unwrap(),
            "{objective}: predict_class"
        );
        // Range predictions: the first half of the iterations.
        let half = model.num_boost_rounds() / 2;
        assert_eq!(
            model.predict_margin_range(&data, ..half).unwrap(),
            gpu.predict_margin_range(&data, ..half).unwrap(),
            "{objective}: predict_margin_range"
        );
    }
}

/// `to_gpu` refuses models that do not predict through the compact forest.
#[test]
fn to_gpu_refuses_unsupported_models() {
    if !device() {
        return;
    }
    let data = dataset(2_000, 5);
    let gblinear = train(
        &TrainingParams::builder()
            .booster(BoosterKind::GbLinear)
            .build()
            .unwrap(),
        &data,
        4,
    )
    .unwrap();
    let err = gblinear.to_gpu().unwrap_err().to_string();
    assert!(err.contains("gblinear"), "{err}");

    let linear = train(
        &TrainingParams::builder()
            .tree_method(TreeMethod::Hist)
            .linear_tree(true)
            .build()
            .unwrap(),
        &data,
        4,
    )
    .unwrap();
    let err = linear.to_gpu().unwrap_err().to_string();
    assert!(err.contains("linear_tree"), "{err}");
}

/// The review's dynamic-range case: in every 128-row slice the first 64
/// rows (bin 0) carry `2^50, 2^26, 2^23 + 1, -2^50, -2^26, -2^23` then zeros
/// and the next 64 (bin 1) the negated sequence. The CPU's `f64` chain sums
/// the bins to +64 and -64; a double-float of `f32`s cannot hold the 51
/// bits, so the backend must take its CPU path.
fn dynamic_range_case() -> (Vec<f32>, Vec<f32>) {
    let six = [
        2f32.powi(50),
        2f32.powi(26),
        2f32.powi(23) + 1.0,
        -(2f32.powi(50)),
        -(2f32.powi(26)),
        -(2f32.powi(23)),
    ];
    let n = 8192;
    let x: Vec<f32> = (0..n).map(|i| f32::from(i % 128 >= 64)).collect();
    let grad: Vec<f32> = (0..n)
        .map(|i| match i % 128 {
            p @ 0..6 => six[p],
            p @ 64..70 => -six[p - 64],
            _ => 0.0,
        })
        .collect();
    (x, grad)
}

/// The histogram of `rows` on `backend` after `prepare(gpair)`, run on one
/// thread so the CPU backend takes its sequential path.
fn histogram(
    backend: &dyn hessboost::internals::HistogramBackend,
    index: &hessboost::internals::GHistIndex,
    rows: &[u32],
    gpair: &[hessboost::objective::GradPair],
) -> Vec<(f64, f64)> {
    let mut out = hessboost::internals::zeroed(index.total_bins());
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            backend.prepare(index, gpair);
            backend.build(index, rows, gpair, &mut out);
        });
    out.iter().map(|s| (s.grad, s.hess)).collect()
}

/// A binned one-feature index of `x`.
fn index_of(x: &[f32]) -> hessboost::internals::GHistIndex {
    let data = DMatrix::from_dense(x, x.len(), 1).unwrap();
    let cuts = hessboost::internals::HistCuts::from_dmatrix(&data, 256);
    hessboost::internals::GHistIndex::from_dmatrix(&data, cuts)
}

/// Gradients outside the double-float's exactness domain give the CPU's
/// histogram bit for bit (the GPU kernels alone returned 0 and 0).
#[test]
fn wide_dynamic_range_histogram_matches_cpu() {
    if !device() {
        return;
    }
    let (x, grad) = dynamic_range_case();
    let index = index_of(&x);
    let gpair: Vec<_> = grad
        .iter()
        .map(|&g| hessboost::objective::GradPair::new(g, 1.0))
        .collect();
    let rows: Vec<u32> = (0..x.len() as u32).collect();
    let cpu = histogram(&hessboost::internals::CpuBackend, &index, &rows, &gpair);
    assert_eq!(cpu, [(64.0, 4096.0), (-64.0, 4096.0)]);
    let backend = metal::MetalHistBackend::new(&index).unwrap();
    assert_eq!(histogram(&backend, &index, &rows, &gpair), cpu);
}

/// The same case through training: with `base_score = 0`, squared error's
/// gradients are the negated labels, and the `device = metal` model is the
/// single-threaded CPU model bit for bit.
#[test]
fn wide_dynamic_range_training_matches_single_threaded_cpu() {
    if !device() {
        return;
    }
    let (x, grad) = dynamic_range_case();
    let labels: Vec<f32> = grad.iter().map(|&g| -g).collect();
    let data = DMatrix::from_dense(&x, x.len(), 1)
        .unwrap()
        .with_labels(&labels)
        .unwrap();
    let build = |device| {
        TrainingParams::builder()
            .objective("reg:squarederror")
            .tree_method(TreeMethod::Hist)
            .base_score(0.0)
            .max_depth(2)
            .device(device)
            .build()
            .unwrap()
    };
    let train_one = |params| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| train(&params, &data, 2).unwrap())
    };
    assert_eq!(
        train_one(build(Device::Cpu)).to_bytes().unwrap(),
        train_one(build(Device::Metal)).to_bytes().unwrap()
    );
}

/// Inputs that do not fit the backend's GPU buffers never reach the GPU: a
/// gradient slice longer than the index and a row list longer than the
/// index (repeated rows) give the CPU's histogram, and a row past the index
/// is refused by the CPU path's bounds check exactly as the CPU backend
/// refuses it, instead of being read past the GPU buffers.
#[test]
fn mismatched_inputs_match_the_cpu_backend() {
    if !device() {
        return;
    }
    let n = 10_000;
    let x: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
    let index = index_of(&x);
    let cpu = hessboost::internals::CpuBackend;
    let backend = metal::MetalHistBackend::new(&index).unwrap();
    let long: Vec<_> = (0..n + 1000)
        .map(|i| hessboost::objective::GradPair::new((i % 7) as f32 - 3.0, 1.0))
        .collect();
    let rows: Vec<u32> = (0..n as u32).collect();
    assert_eq!(
        histogram(&backend, &index, &rows, &long),
        histogram(&cpu, &index, &rows, &long)
    );
    let gpair = &long[..n];
    let twice: Vec<u32> = rows.iter().chain(&rows).copied().collect();
    assert_eq!(
        histogram(&backend, &index, &twice, gpair),
        histogram(&cpu, &index, &twice, gpair)
    );
    let past_end: Vec<u32> = (1..=n as u32).collect();
    let refused = |backend: &dyn hessboost::internals::HistogramBackend| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            histogram(backend, &index, &past_end, gpair)
        }))
        .is_err()
    };
    assert!(refused(&cpu));
    assert!(refused(&backend));
}
