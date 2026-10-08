//! The integration suite every GPU backend runs: each backend's test file
//! implements [`GpuBackend`] and calls the suite's functions from `#[test]`
//! wrappers. Tests that need the backend skip when it cannot run here
//! (printing why); parameter tests always run.

use hessboost::config::{
    BoosterKind, Dart, Device, LinearTree, ProcessType, QuantizedGrad, Refresh,
};
use hessboost::data::FeatureType;
use hessboost::internals::{CpuBackend, GHistIndex, HistCuts, HistogramBackend, zeroed};
use hessboost::model::Predictions;
use hessboost::objective::{GradPair, Multiclass};
use hessboost::prelude::*;

use super::{incompatible_model, invalid_param, with_threads};

/// A GPU backend under test.
pub trait GpuBackend {
    /// Name in skip messages and the `device` XGBoost spells it as (`cuda`,
    /// `metal`, `wgpu`).
    const NAME: &'static str;
    /// The `device` that trains on this backend.
    const DEVICE: Device;
    /// Rows of a prediction batch spanning several of the backend's
    /// prediction blocks.
    const MULTI_BLOCK_ROWS: usize;
    /// The backend's GPU predictor.
    type Model: GpuPredictor;
    /// Why the backend cannot run here, `None` when it can.
    fn unavailable_reason() -> Option<String>;
    /// The backend's histogram builder for `index`.
    fn hist_backend(index: &GHistIndex) -> Box<dyn HistogramBackend>;
    /// `model` laid out for GPU prediction (`to_cuda`, `to_gpu`, `to_wgpu`).
    fn to_gpu(model: &BoostedModel) -> Result<Self::Model>;
}

/// The prediction methods every GPU predictor shares with `BoostedModel`
/// (`Debug` for the refusal checks' error messages).
pub trait GpuPredictor: std::fmt::Debug {
    /// As [`BoostedModel::predict`].
    fn predict(&self, data: &DMatrix, iterations: Iterations) -> Result<Predictions>;
    /// As [`BoostedModel::predict_margin`].
    fn predict_margin(&self, data: &DMatrix, iterations: Iterations) -> Result<Predictions>;
    /// As [`BoostedModel::predict_class`].
    fn predict_class(&self, data: &DMatrix, iterations: Iterations) -> Result<Predictions<u32>>;
}

/// Whether backend `B` can run here, with the skip reason printed so a
/// vacuous pass is visible.
pub fn available<B: GpuBackend>() -> bool {
    if let Some(reason) = B::unavailable_reason() {
        eprintln!("skipping {} test: {reason}", B::NAME);
        return false;
    }
    true
}

/// A deterministic regression dataset with missing values and a categorical
/// first column.
pub fn dataset(n: usize, cols: usize) -> DMatrix {
    dataset_with(n, cols, true, true)
}

/// [`dataset`] with or without its missing values, and with its first column
/// categorical or numeric.
pub fn dataset_with(n: usize, cols: usize, missing: bool, categorical: bool) -> DMatrix {
    let mut x = vec![0.0f32; n * cols];
    let mut y = vec![0.0f32; n];
    for r in 0..n {
        let mut target = 0.0;
        for f in 0..cols {
            let v = if f == 0 {
                ((r * 31 + f) % 5) as f32 // five codes, categories if `categorical`
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
    let types: Vec<FeatureType> = (0..cols)
        .map(|f| {
            if f == 0 && categorical {
                FeatureType::Categorical
            } else {
                FeatureType::Numerical
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

/// Training on `B` reproduces single-threaded CPU training bit for bit,
/// tree for tree (the whole serialized model compares equal), SGLB
/// posterior sampling included (its noise is drawn on the CPU and its
/// leaves re-estimated there).
pub fn training_matches_single_threaded_cpu<B: GpuBackend>() {
    if !available::<B>() {
        return;
    }
    let data = dataset(40_000, 12);
    for posterior_sampling in [false, true] {
        let build = |device| {
            TrainingParams::builder()
                .objective(Objective::SquaredError(RegLoss::default()))
                .tree_method(TreeMethod::Hist)
                .max_depth(6)
                .eta(0.3)
                .posterior_sampling(posterior_sampling)
                .device(device)
                .build()
                .unwrap()
        };
        let train_one = |params| with_threads(1, || train(&params, &data, 10).unwrap());
        let cpu = train_one(build(Device::Cpu));
        let gpu = train_one(build(B::DEVICE));
        assert_eq!(
            cpu.encode(ModelFormat::Binary).unwrap(),
            gpu.encode(ModelFormat::Binary).unwrap(),
            "the {}-trained model must be bit-identical to the CPU's \
             (posterior sampling {posterior_sampling})",
            B::NAME
        );
    }
}

/// A run on `B` repeats itself exactly, independent of the worker count,
/// and equals the multi-threaded CPU run.
pub fn training_is_deterministic<B: GpuBackend>() {
    if !available::<B>() {
        return;
    }
    let data = dataset(20_000, 9);
    let build = |device| {
        TrainingParams::builder()
            .objective(Objective::BinaryLogistic(RegLoss::default()))
            .tree_method(TreeMethod::Hist)
            .max_depth(6)
            .eta(0.3)
            .subsample(0.8)
            .device(device)
            .build()
            .unwrap()
    };
    let labels: Vec<f32> = data
        .labels()
        .unwrap()
        .iter()
        .map(|&y| f32::from(y >= 1.5))
        .collect();
    let data = data.with_labels(&labels).unwrap();
    let run = |threads, device| {
        let params = build(device);
        with_threads(threads, || {
            train(&params, &data, 8)
                .unwrap()
                .encode(ModelFormat::Binary)
                .unwrap()
        })
    };
    assert_eq!(run(1, B::DEVICE), run(1, B::DEVICE));
    assert_eq!(run(1, B::DEVICE), run(4, B::DEVICE));
    assert_eq!(run(4, B::DEVICE), run(4, Device::Cpu));
}

/// The unsupported combinations with `B`'s `device` are refused with an
/// error, never silently ignored.
pub fn refuses_unsupported_combinations<B: GpuBackend>() {
    let base = TrainingParams::builder().device(B::DEVICE).build().unwrap();
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
        assert_eq!(invalid_param(params.validate()), "device", "{name}");
    }
}

/// `B`'s `device` round-trips through XGBoost's flat parameters as its
/// name.
pub fn round_trips_through_xgboost_params<B: GpuBackend>() {
    let params = TrainingParams::builder().device(B::DEVICE).build().unwrap();
    let flat = params.to_xgboost().unwrap();
    assert_eq!(flat.get("device").and_then(|v| v.as_str()), Some(B::NAME));
    let back = TrainingParams::from_xgboost(flat).unwrap();
    assert_eq!(back.device, B::DEVICE);
}

/// GPU predictions are bit-identical to the model's across objectives,
/// missing values, categorical splits, DART weights, and iteration ranges.
pub fn predicts_bit_identically<B: GpuBackend>() {
    if !available::<B>() {
        return;
    }
    let cases = [
        Objective::SquaredError(RegLoss::default()),
        Objective::BinaryLogistic(RegLoss::default()),
        Objective::Softmax(Multiclass::new(4).unwrap()),
    ];
    for spec in cases {
        let objective = spec.name();
        let num_class = spec.num_class().unwrap_or(0);
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
        let params = TrainingParams::builder()
            .objective(spec.clone())
            .tree_method(TreeMethod::Hist)
            .max_depth(5)
            .eta(0.4)
            .booster(BoosterKind::Dart(Dart::default()))
            .build()
            .unwrap();
        let model = train(&params, &data, 15).unwrap();
        let gpu = B::to_gpu(&model).unwrap();
        assert_eq!(
            model.predict(&data, Iterations::Best).unwrap(),
            gpu.predict(&data, Iterations::Best).unwrap(),
            "{objective}: predict"
        );
        assert_eq!(
            model.predict_margin(&data, Iterations::Best).unwrap(),
            gpu.predict_margin(&data, Iterations::Best).unwrap(),
            "{objective}: predict_margin"
        );
        assert_eq!(
            model.predict_class(&data, Iterations::Best).unwrap(),
            gpu.predict_class(&data, Iterations::Best).unwrap(),
            "{objective}: predict_class"
        );
        // Range predictions: the first half of the iterations.
        let half = model.num_boost_rounds() / 2;
        assert_eq!(
            model.predict_margin(&data, ..half).unwrap(),
            gpu.predict_margin(&data, Iterations::from(..half)).unwrap(),
            "{objective}: predict_margin(..half)"
        );
    }
}

/// A batch of [`GpuBackend::MULTI_BLOCK_ROWS`] rows, larger than one
/// prediction block: the GPU runs the call block by block, and every block
/// still lands bit-identical to the CPU's single walk, for one output (the
/// regression model) and for several (the multiclass model).
pub fn predicts_bit_identically_across_blocks<B: GpuBackend>() {
    if !available::<B>() {
        return;
    }
    let cols = 5;
    let train_data = dataset(4_000, cols);
    let batch = dataset(B::MULTI_BLOCK_ROWS, cols);
    for spec in [
        Objective::SquaredError(RegLoss::default()),
        Objective::Softmax(Multiclass::new(3).unwrap()),
    ] {
        // A multiclass objective needs labels inside its class range.
        let data = match spec.num_class() {
            Some(classes) => {
                let labels: Vec<f32> = train_data
                    .labels()
                    .unwrap()
                    .iter()
                    .map(|&y| y.trunc().abs() % classes as f32)
                    .collect();
                train_data.clone().with_labels(&labels).unwrap()
            }
            None => train_data.clone(),
        };
        let objective = spec.name().to_owned();
        let params = TrainingParams::builder()
            .objective(spec)
            .tree_method(TreeMethod::Hist)
            .max_depth(4)
            .eta(0.4)
            .build()
            .unwrap();
        let model = train(&params, &data, 8).unwrap();
        let gpu = B::to_gpu(&model).unwrap();
        assert_eq!(
            model.predict_margin(&batch, Iterations::Best).unwrap(),
            gpu.predict_margin(&batch, Iterations::Best).unwrap(),
            "{objective}: predict_margin across blocks"
        );
        assert_eq!(
            model.predict(&batch, Iterations::Best).unwrap(),
            gpu.predict(&batch, Iterations::Best).unwrap(),
            "{objective}: predict across blocks"
        );
    }
}

/// GPU prediction refuses models that do not predict through the compact
/// forest.
pub fn refuses_unsupported_models<B: GpuBackend>() {
    if !available::<B>() {
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
    assert_eq!(incompatible_model(B::to_gpu(&gblinear)), "model");

    let linear = train(
        &TrainingParams::builder()
            .tree_method(TreeMethod::Hist)
            .linear_tree(LinearTree::default())
            .build()
            .unwrap(),
        &data,
        4,
    )
    .unwrap();
    assert_eq!(incompatible_model(B::to_gpu(&linear)), "model");
}

/// The dynamic-range case: in every 128-row slice the first 64 rows (bin 0)
/// carry `2^50, 2^26, 2^23 + 1, -2^50, -2^26, -2^23` then zeros and the next
/// 64 (bin 1) the negated sequence. The CPU's `f64` chain sums the bins to
/// +64 and -64 (the first double-float kernels returned 0 and 0). Values up
/// to `2^50` with a grain of 1 are exact on the GPU for at most 8 rows per
/// node, so the backend must take its CPU path.
pub fn dynamic_range_case() -> (Vec<f32>, Vec<f32>) {
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
pub fn histogram(
    backend: &dyn HistogramBackend,
    index: &GHistIndex,
    rows: &[u32],
    gpair: &[GradPair],
) -> Vec<(f64, f64)> {
    let mut out = zeroed(index.total_bins());
    with_threads(1, || {
        backend.prepare(index, gpair);
        backend.build(index, rows, gpair, &mut out);
    });
    out.iter().map(|s| (s.grad, s.hess)).collect()
}

/// A binned one-feature index of `x`.
pub fn index_of(x: &[f32]) -> GHistIndex {
    let data = DMatrix::from_dense(x, x.len(), 1).unwrap();
    let cuts = HistCuts::from_dmatrix(&data, 256);
    GHistIndex::from_dmatrix(&data, cuts)
}

/// Gradients outside the GPU's exactness bound give the CPU's histogram
/// bit for bit.
pub fn wide_dynamic_range_histogram_matches_cpu<B: GpuBackend>() {
    if !available::<B>() {
        return;
    }
    let (x, grad) = dynamic_range_case();
    let index = index_of(&x);
    let gpair: Vec<_> = grad.iter().map(|&g| GradPair::new(g, 1.0)).collect();
    let rows: Vec<u32> = (0..x.len() as u32).collect();
    let cpu = histogram(&CpuBackend, &index, &rows, &gpair);
    assert_eq!(cpu, [(64.0, 4096.0), (-64.0, 4096.0)]);
    let backend = B::hist_backend(&index);
    assert_eq!(histogram(backend.as_ref(), &index, &rows, &gpair), cpu);
}

/// The same case through training: with `base_score = 0`, squared error's
/// gradients are the negated labels, and the model trained on `B` is the
/// single-threaded CPU model bit for bit.
pub fn wide_dynamic_range_training_matches_single_threaded_cpu<B: GpuBackend>() {
    if !available::<B>() {
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
            .objective(Objective::SquaredError(RegLoss::default()))
            .tree_method(TreeMethod::Hist)
            .base_score(0.0)
            .max_depth(2)
            .device(device)
            .build()
            .unwrap()
    };
    let train_one = |params| with_threads(1, || train(&params, &data, 2).unwrap());
    assert_eq!(
        train_one(build(Device::Cpu))
            .encode(ModelFormat::Binary)
            .unwrap(),
        train_one(build(B::DEVICE))
            .encode(ModelFormat::Binary)
            .unwrap()
    );
}

/// Inputs that do not fit the backend's GPU buffers never reach the GPU: a
/// gradient slice longer than the index and a row list longer than the
/// index (repeated rows) give the CPU's histogram, and a row past the index
/// is refused by the CPU path's bounds check exactly as the CPU backend
/// refuses it, instead of being read past the GPU buffers.
pub fn mismatched_inputs_match_the_cpu_backend<B: GpuBackend>() {
    if !available::<B>() {
        return;
    }
    let n = 10_000;
    let x: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
    let index = index_of(&x);
    let cpu = CpuBackend;
    let backend = B::hist_backend(&index);
    let long: Vec<_> = (0..n + 1000)
        .map(|i| GradPair::new((i % 7) as f32 - 3.0, 1.0))
        .collect();
    let rows: Vec<u32> = (0..n as u32).collect();
    assert_eq!(
        histogram(backend.as_ref(), &index, &rows, &long),
        histogram(&cpu, &index, &rows, &long)
    );
    let gpair = &long[..n];
    let twice: Vec<u32> = rows.iter().chain(&rows).copied().collect();
    assert_eq!(
        histogram(backend.as_ref(), &index, &twice, gpair),
        histogram(&cpu, &index, &twice, gpair)
    );
    let past_end: Vec<u32> = (1..=n as u32).collect();
    let refused = |backend: &dyn HistogramBackend| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            histogram(backend, &index, &past_end, gpair)
        }))
        .is_err()
    };
    assert!(refused(&cpu));
    assert!(refused(backend.as_ref()));
}
