//! Criterion benchmarks for histogram construction, objective and metric
//! kernels, prediction transforms, end-to-end training (including the opt-in
//! growers and eval sets), prediction and SHAP, model serialization, and
//! data preparation.

use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use hessboost::config::{BoosterKind, GrowPolicy, Monotone, MultiStrategy, TrainingParamsBuilder};
use hessboost::data::FeatureType;
use hessboost::internals::{
    ColumnSampler, CpuBackend, GHistIndex, HistCuts, HistTreeBuilder, HistogramBackend, zeroed,
};
use hessboost::metric::{
    ErrorRate, GammaNLogLik, LogLoss, Mae, Metric, PoissonNLogLik, Rmse, create_metric,
};
use hessboost::objective::{
    Gamma, GradPair, Logistic, Objective, Poisson, Softmax, Tweedie, create_objective,
};
use hessboost::prelude::*;
use hessboost::training::budget::{BudgetConfig, train_with_budget};
use std::hint::black_box;

/// Deterministic synthetic regression dataset.
fn make_data(n: usize, f: usize) -> DMatrix {
    let mut x = vec![0f32; n * f];
    let mut y = vec![0f32; n];
    let mut s: u64 = 0x2545_F491_4F6C_DD1D;
    let mut rng = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 33) as f32) / (1u32 << 31) as f32
    };
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..f {
            let v = rng();
            x[i * f + j] = v;
            if j < 5 {
                acc += v * (j as f32 + 1.0);
            }
        }
        y[i] = acc + rng() * 0.1;
    }
    DMatrix::from_dense(&x, n, f)
        .unwrap()
        .with_labels(&y)
        .unwrap()
}

fn make_binary_data(n: usize, f: usize) -> DMatrix {
    let data = make_data(n, f);
    let labels: Vec<f32> = data
        .labels()
        .unwrap()
        .iter()
        .map(|&y| f32::from(y >= 7.5))
        .collect();
    data.with_labels(&labels).unwrap()
}

fn make_weights(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 + (i % 17) as f32 * 0.0625).collect()
}

/// `make_data`'s values as a CSR matrix keeping about 40% of the entries
/// (the others are absent), with the same labels.
fn make_csr_data(n: usize, f: usize) -> DMatrix {
    let data = make_data(n, f);
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0);
    let (mut indices, mut values) = (Vec::new(), Vec::new());
    for i in 0..n {
        for j in 0..f {
            if (i * 7 + j * 3) % 5 < 2 {
                indices.push(j as u32);
                values.push(data.get(i, j).unwrap());
            }
        }
        indptr.push(indices.len());
    }
    DMatrix::from_csr(indptr, indices, values, f)
        .unwrap()
        .with_labels(data.labels().unwrap())
        .unwrap()
}

/// `make_data` with its first two features recoded as categories: 3 (below
/// `max_cat_to_onehot`, one-hot splits) and 24 (partition splits).
fn make_categorical_data(n: usize, f: usize) -> DMatrix {
    let data = make_data(n, f);
    let mut types = vec![FeatureType::Numerical; f];
    types[..2].fill(FeatureType::Categorical);
    let values: Vec<f32> = (0..n * f)
        .map(|i| {
            let v = data.get(i / f, i % f).unwrap();
            match i % f {
                0 => (v * 3.0).floor(),
                1 => (v * 24.0).floor(),
                _ => v,
            }
        })
        .collect();
    DMatrix::from_dense(&values, n, f)
        .unwrap()
        .with_labels(data.labels().unwrap())
        .unwrap()
        .with_feature_types(&types)
        .unwrap()
}

/// A binned index of `data` (256 bins), synthetic gradient pairs, and every
/// row: the inputs of one root histogram build.
fn histogram_case(data: &DMatrix) -> (GHistIndex, Vec<GradPair>, Vec<u32>) {
    let n = data.n_rows();
    let cuts = HistCuts::from_dmatrix(data, 256);
    let ghist = GHistIndex::from_dmatrix(data, cuts);
    let gpair = (0..n)
        .map(|i| GradPair::new((i % 7) as f32 - 3.0, 1.0))
        .collect();
    (ghist, gpair, (0..n as u32).collect())
}

/// A feature-less `n`-row matrix carrying only metadata, for benches that
/// evaluate objectives and metrics through `MetaInfo`.
fn meta_rows(n: usize) -> DMatrix {
    DMatrix::from_dense(&vec![0.0; n], n, 1).unwrap()
}

/// Query sizes splitting `n` rows into groups of `size`.
fn group_sizes(n: usize, size: usize) -> Vec<usize> {
    let mut sizes = vec![size; n / size];
    if !n.is_multiple_of(size) {
        sizes.push(n % size);
    }
    sizes
}

/// Graded relevance labels (0-4) for the ranking benches.
fn relevance_labels(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 7) % 13 % 5) as f32).collect()
}

/// The 1M-element scale used by the per-element kernel benches.
const N: usize = 1_000_000;

/// Predictions sweeping a wide range of margins.
fn wide_range(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 2_001) as f32 * 0.005 - 5.0).collect()
}

/// 0/1 labels alternating by row.
fn alternating_labels(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 2) as f32).collect()
}

/// Strictly positive labels for the count objectives/metrics.
fn positive_labels(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.25 + (i % 101) as f32 * 0.02).collect()
}

/// Emit the unweighted/weighted bench pair for one metric.
fn bench_metric_pair(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    metric: &dyn Metric,
    preds: &[f32],
    labels: &[f32],
    weights: &[f32],
) {
    group.bench_function(format!("{name}_unweighted_1m"), |b| {
        b.iter(|| black_box(metric.eval(preds, labels, None)));
    });
    group.bench_function(format!("{name}_weighted_1m"), |b| {
        b.iter(|| black_box(metric.eval(preds, labels, Some(weights))));
    });
}

fn scalar_sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn bench_histogram_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("histogram_build");
    for &n in &[10_000usize, 100_000] {
        let (ghist, gpair, rows) = histogram_case(&make_data(n, 30));
        let mut out = zeroed(ghist.total_bins());

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| CpuBackend.build(&ghist, &rows, &gpair, &mut out));
        });
    }
    // A sparse (CSR) index: row-wise accumulation.
    let n = 100_000;
    let (ghist, gpair, rows) = histogram_case(&make_csr_data(n, 30));
    let mut out = zeroed(ghist.total_bins());
    group.throughput(Throughput::Elements(n as u64));
    group.bench_with_input(BenchmarkId::new("sparse", n), &n, |b, _| {
        b.iter(|| CpuBackend.build(&ghist, &rows, &gpair, &mut out));
    });
    group.finish();
}

fn bench_hist_tree_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("hist_tree_build");
    // `quantized` adds a `{name}_quantized` run with opt-in quantized
    // gradients (`use_quantized_grad`) on the same data.
    for (name, n, features, depth, quantized) in [
        ("depth1", 50_000, 20, 1, false),
        ("depth6", 50_000, 20, 6, true),
        ("depth10", 50_000, 20, 10, true),
        ("wide128", 10_000, 128, 6, true),
        ("missing", 50_000, 20, 6, true),
        ("monotone", 50_000, 20, 6, false),
        ("lossguide", 50_000, 20, 6, false),
        ("large_depth8", 1_000_000, 50, 8, true),
    ] {
        let mut data = make_data(n, features);
        if name == "missing" {
            let values: Vec<f32> = (0..n * features)
                .map(|i| {
                    if i % 11 < 2 {
                        f32::NAN
                    } else {
                        data.get(i / features, i % features).unwrap()
                    }
                })
                .collect();
            data = DMatrix::from_dense(&values, n, features)
                .unwrap()
                .with_labels(data.labels().unwrap())
                .unwrap();
        }
        let cuts = HistCuts::from_dmatrix(&data, 256);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        let gpair: Vec<GradPair> = data
            .labels()
            .unwrap()
            .iter()
            .map(|&label| GradPair::new(7.5 - label, 1.0))
            .collect();
        let rows: Vec<u32> = (0..n as u32).collect();
        // A 1M-row tree takes long enough that ten samples are stable.
        group.sample_size(if n >= 1_000_000 { 10 } else { 100 });
        let variants: &[bool] = if quantized { &[false, true] } else { &[false] };
        for &use_quantized_grad in variants {
            let params = TrainingParams::builder()
                .max_depth(depth)
                .grow_policy(if name == "lossguide" {
                    GrowPolicy::LossGuide
                } else {
                    GrowPolicy::DepthWise
                })
                .max_leaves(64)
                .monotone_constraints(if name == "monotone" {
                    vec![Monotone::Increasing]
                } else {
                    Vec::new()
                })
                .use_quantized_grad(use_quantized_grad)
                .build()
                .unwrap();
            let builder = HistTreeBuilder::new(&params);
            let id = if use_quantized_grad {
                format!("{name}_quantized")
            } else {
                name.to_owned()
            };
            group.bench_function(id, |b| {
                b.iter(|| {
                    let mut sampler = ColumnSampler::all(features);
                    black_box(builder.build(&ghist, &gpair, &rows, &mut sampler))
                });
            });
        }
    }
    group.finish();
}

fn bench_objective_gradients(c: &mut Criterion) {
    let preds: Vec<f32> = wide_range(N);
    let labels: Vec<f32> = alternating_labels(N);
    let positive_labels: Vec<f32> = positive_labels(N);
    let weights: Vec<f32> = make_weights(N);
    let mut out = vec![GradPair::default(); N];
    let mut group = c.benchmark_group("objective_gradient");
    group.throughput(Throughput::Elements(N as u64));

    let mut run = |name: &str, objective: &dyn Objective, y: &[f32], weights: Option<&[f32]>| {
        group.bench_function(name, |b| {
            b.iter(|| {
                objective.gradient(&preds, y, weights, &mut out);
                black_box(&out);
            });
        });
    };
    run(
        "logistic_unweighted_1m",
        &Logistic::default(),
        &labels,
        None,
    );
    run(
        "logistic_weighted_1m",
        &Logistic::new(1.5),
        &labels,
        Some(&weights),
    );
    run(
        "poisson_unweighted_1m",
        &Poisson::default(),
        &positive_labels,
        None,
    );
    run(
        "gamma_unweighted_1m",
        &Gamma::default(),
        &positive_labels,
        None,
    );
    run(
        "tweedie_unweighted_1m",
        &Tweedie::default(),
        &positive_labels,
        None,
    );
    group.finish();

    let mut group = c.benchmark_group("objective_gradient_multiclass");
    for k in [2usize, 3, 4, 8, 16, 24, 32, 128] {
        let rows = N / k;
        let multi_preds: Vec<f32> = (0..rows * k)
            .map(|i| (i % 101) as f32 * 0.025 - 1.25)
            .collect();
        let multi_labels: Vec<f32> = (0..rows).map(|i| (i % k) as f32).collect();
        let multi_weights: Vec<f32> = make_weights(rows);
        let mut multi_out = vec![GradPair::default(); rows * k];
        let softmax = Softmax::new(k, true);
        group.throughput(Throughput::Elements((rows * k) as u64));
        for (suffix, weights) in [("", None), ("_weighted", Some(multi_weights.as_slice()))] {
            group.bench_function(format!("softmax_k{k}{suffix}_1m_outputs"), |b| {
                b.iter(|| {
                    softmax.gradient(&multi_preds, &multi_labels, weights, &mut multi_out);
                    black_box(&multi_out);
                });
            });
        }
    }
    group.finish();
}

fn bench_prediction_transforms(c: &mut Criterion) {
    let source: Vec<f32> = wide_range(N);
    let mut values = source.clone();
    let mut group = c.benchmark_group("prediction_transform");
    group.throughput(Throughput::Elements(N as u64));

    group.bench_function("logistic_automatic", |b| {
        b.iter(|| {
            values.copy_from_slice(&source);
            Logistic::default().pred_transform(&mut values);
            black_box(&values);
        });
    });
    group.bench_function("logistic_scalar_reference", |b| {
        b.iter(|| {
            values.copy_from_slice(&source);
            for value in &mut values {
                *value = scalar_sigmoid(*value);
            }
            black_box(&values);
        });
    });
    group.bench_function("exp_automatic", |b| {
        b.iter(|| {
            values.copy_from_slice(&source);
            Gamma::default().pred_transform(&mut values);
            black_box(&values);
        });
    });
    group.bench_function("exp_scalar_reference", |b| {
        b.iter(|| {
            values.copy_from_slice(&source);
            for value in &mut values {
                *value = value.exp();
            }
            black_box(&values);
        });
    });
    group.finish();

    let mut group = c.benchmark_group("prediction_transform_multiclass");
    for num_class in [2, 3, 4, 8, 17, 32, 128] {
        let len = N / num_class * num_class;
        let source = &source[..len];
        let mut values = source.to_vec();
        let objective = Softmax::new(num_class, true);
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(format!("softmax_k{num_class}_1m_outputs"), |b| {
            b.iter(|| {
                values.copy_from_slice(source);
                objective.pred_transform(&mut values);
                black_box(&values);
            });
        });
    }
    group.finish();
}

fn bench_pointwise_metrics(c: &mut Criterion) {
    let preds: Vec<f32> = (0..N).map(|i| (i % 1_001) as f32 * 0.001).collect();
    let labels: Vec<f32> = alternating_labels(N);
    let weights: Vec<f32> = make_weights(N);
    let mut group = c.benchmark_group("pointwise_metric");
    group.throughput(Throughput::Elements(N as u64));

    for (name, metric) in [
        ("rmse", &Rmse::default() as &dyn Metric),
        ("mae", &Mae::default() as &dyn Metric),
        ("error", &ErrorRate::default() as &dyn Metric),
    ] {
        bench_metric_pair(&mut group, name, metric, &preds, &labels, &weights);
    }
    group.finish();
}

fn bench_log_metrics(c: &mut Criterion) {
    let probabilities: Vec<f32> = (0..N).map(|i| 0.001 + (i % 999) as f32 * 0.001).collect();
    let binary_labels: Vec<f32> = alternating_labels(N);
    let positive_labels: Vec<f32> = positive_labels(N);
    let weights: Vec<f32> = make_weights(N);
    let mut group = c.benchmark_group("log_metric");
    group.throughput(Throughput::Elements(N as u64));

    let tweedie = create_metric("tweedie-nloglik@1.5", &TrainingParams::default()).unwrap();
    for (name, metric, labels) in [
        (
            "logloss",
            &LogLoss::default() as &dyn Metric,
            binary_labels.as_slice(),
        ),
        (
            "poisson_nloglik",
            &PoissonNLogLik::default() as &dyn Metric,
            positive_labels.as_slice(),
        ),
        (
            "gamma_nloglik",
            &GammaNLogLik::default() as &dyn Metric,
            positive_labels.as_slice(),
        ),
        (
            "tweedie_nloglik",
            tweedie.as_ref(),
            positive_labels.as_slice(),
        ),
    ] {
        bench_metric_pair(&mut group, name, metric, &probabilities, labels, &weights);
    }
    group.finish();
}

fn bench_multiclass_metrics(c: &mut Criterion) {
    let outputs = N;
    let mut group = c.benchmark_group("multiclass_metric");
    for num_class in [4usize, 8, 32, 128] {
        let rows = outputs / num_class;
        let probabilities: Vec<f32> = (0..rows * num_class)
            .map(|index| 0.001 + (index % 999) as f32 * 0.001)
            .collect();
        let labels: Vec<f32> = (0..rows)
            .map(|row| ((row * 7) % num_class) as f32)
            .collect();
        let weights: Vec<f32> = make_weights(rows);
        group.throughput(Throughput::Elements(rows as u64));
        let params = TrainingParams::builder()
            .objective("multi:softprob")
            .num_class(num_class)
            .build()
            .unwrap();
        for metric_name in ["mlogloss", "merror"] {
            let metric = create_metric(metric_name, &params).unwrap();
            group.bench_function(format!("{metric_name}_k{num_class}_unweighted"), |b| {
                b.iter(|| black_box(metric.eval(&probabilities, &labels, None)));
            });
            group.bench_function(format!("{metric_name}_k{num_class}_weighted"), |b| {
                b.iter(|| black_box(metric.eval(&probabilities, &labels, Some(&weights))));
            });
        }
    }
    group.finish();
}

fn scalar_logistic_objective(base_margin: f32) -> hessboost::objective::CustomObjective {
    hessboost::objective::CustomObjective::new(
        "scalar:logistic",
        1,
        base_margin,
        "logloss",
        |preds, labels, weights, out| {
            for i in 0..preds.len() {
                let probability = scalar_sigmoid(preds[i]);
                let weight = weights.map_or(1.0, |values| values[i]);
                out[i] = GradPair::new(
                    (probability - labels[i]) * weight,
                    (probability * (1.0 - probability)).max(1e-16) * weight,
                );
            }
        },
    )
}

fn bench_binary_train(c: &mut Criterion) {
    let data = make_binary_data(50_000, 20);
    let labels = data.labels().unwrap();
    let positive_rate =
        labels.iter().map(|&label| f64::from(label)).sum::<f64>() / labels.len() as f64;
    let base_margin = (positive_rate / (1.0 - positive_rate)).ln() as f32;
    let params = TrainingParams::builder()
        .objective("binary:logistic")
        .tree_method(TreeMethod::Hist)
        .max_depth(6)
        .eta(0.1)
        .build()
        .unwrap();
    let mut group = c.benchmark_group("train_binary_50k_x20_50rounds");
    group.sample_size(10);
    group.bench_function("automatic_dispatch", |b| {
        b.iter(|| black_box(train(&params, &data, 50).unwrap()));
    });
    group.bench_function("scalar_objective_reference", |b| {
        b.iter(|| {
            black_box(
                Trainer::new(&params, &data, 50)
                    .objective(&scalar_logistic_objective(base_margin))
                    .train()
                    .unwrap()
                    .model,
            )
        });
    });
    let mut quantized = params.clone();
    quantized.use_quantized_grad = true;
    group.bench_function("quantized", |b| {
        b.iter(|| black_box(train(&quantized, &data, 50).unwrap()));
    });
    group.finish();
}

fn bench_train(c: &mut Criterion) {
    let data = make_data(50_000, 20);
    let mut group = c.benchmark_group("train_50k_x20_50rounds");
    group.sample_size(10);

    for (name, method, alpha, max_bin, quantized) in [
        ("Hist", TreeMethod::Hist, 0.0, 256, false),
        ("Exact", TreeMethod::Exact, 0.0, 256, false),
        ("Hist_l1", TreeMethod::Hist, 1.0, 256, false),
        ("Hist_16bins", TreeMethod::Hist, 0.0, 16, false),
        ("Hist_quantized", TreeMethod::Hist, 0.0, 256, true),
    ] {
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .tree_method(method)
            .max_depth(6)
            .eta(0.1)
            .alpha(alpha)
            .max_bin(max_bin)
            .use_quantized_grad(quantized)
            .build()
            .unwrap();
        group.bench_function(name, |b| {
            b.iter(|| train(&params, &data, 50).unwrap());
        });
    }
    group.finish();
}

/// Batch prediction on 100k x 30 rows: a symmetric model (bit-pattern
/// tables) against a depthwise model of the same depth and size (generic
/// lockstep walk).
fn bench_predict(c: &mut Criterion) {
    let data = make_data(100_000, 30);
    let mut group = c.benchmark_group("predict_100k_x30_100trees_depth6");
    group.sample_size(20);
    group.throughput(Throughput::Elements(data.n_rows() as u64));
    for (name, policy) in [
        ("depthwise", GrowPolicy::DepthWise),
        ("symmetric", GrowPolicy::Symmetric),
    ] {
        let params = TrainingParams::builder()
            .tree_method(TreeMethod::Hist)
            .grow_policy(policy)
            .max_depth(6)
            .eta(0.1)
            .build()
            .unwrap();
        let model = train(&params, &data, 100).unwrap();
        group.bench_function(name, |b| {
            b.iter(|| model.predict_margin(black_box(&data)).unwrap());
        });
    }
    group.finish();
}

/// QuadratureTreeSHAP on 100 depth-6 trees over 20 features (trained on 20k
/// rows): contributions for 2,000 rows and interaction values for 200.
fn bench_shap(c: &mut Criterion) {
    let data = make_data(20_000, 20);
    let params = TrainingParams::builder()
        .tree_method(TreeMethod::Hist)
        .max_depth(6)
        .eta(0.1)
        .build()
        .unwrap();
    let model = train(&params, &data, 100).unwrap();
    let values: Vec<f32> = (0..2_000 * 20)
        .map(|i| ((i * 7919) % 1000) as f32 / 1000.0)
        .collect();
    let rows = DMatrix::from_dense(&values, 2_000, 20).unwrap();
    let few_rows = DMatrix::from_dense(&values[..200 * 20], 200, 20).unwrap();
    let mut group = c.benchmark_group("shap_x20_100trees_depth6");
    group.sample_size(10);
    group.bench_function("contribs_2k", |b| {
        b.iter(|| model.predict_contribs(black_box(&rows)).unwrap());
    });
    group.bench_function("interactions_200", |b| {
        b.iter(|| model.predict_interactions(black_box(&few_rows)).unwrap());
    });
    group.finish();
}

/// Gradients of the objectives the pointwise kernels do not cover, through
/// `gradient_info` (the training path). Alpha lists and label matrices keep
/// the output count at about one million, like the multiclass benches.
fn bench_other_gradients(c: &mut Criterion) {
    let mut group = c.benchmark_group("objective_gradient_other");
    let objective = |b: TrainingParamsBuilder, n_targets: usize| {
        create_objective(&b.build().unwrap(), n_targets).unwrap()
    };
    let margins =
        |len: usize| -> Vec<f32> { (0..len).map(|i| (i % 1_001) as f32 * 0.004 - 2.0).collect() };
    let regression_labels =
        |n: usize| -> Vec<f32> { (0..n).map(|i| (i % 997) as f32 * 0.004 - 2.0).collect() };
    let mut run = |name: &str, objective: &dyn Objective, data: &DMatrix, width: usize| {
        let n = data.n_rows();
        let preds = margins(n * width);
        let mut out = vec![GradPair::default(); n * width];
        let info = data.info();
        group.throughput(Throughput::Elements((n * width) as u64));
        group.bench_function(name, |b| {
            b.iter(|| {
                objective.gradient_info(&preds, &info, &mut out);
                black_box(&out);
            });
        });
    };
    let base = TrainingParams::builder;
    let rows = meta_rows(N).with_labels(&regression_labels(N)).unwrap();
    run(
        "absoluteerror_1m",
        objective(base().objective("reg:absoluteerror"), 1).as_ref(),
        &rows,
        1,
    );
    let weighted = meta_rows(N)
        .with_labels(&regression_labels(N))
        .unwrap()
        .with_weights(&make_weights(N))
        .unwrap();
    run(
        "absoluteerror_weighted_1m",
        objective(base().objective("reg:absoluteerror"), 1).as_ref(),
        &weighted,
        1,
    );
    let k = 3;
    let matrix = meta_rows(N / k)
        .with_label_matrix(&regression_labels(N / k * k), k)
        .unwrap();
    run(
        "absoluteerror_k3_1m_outputs",
        objective(base().objective("reg:absoluteerror"), k).as_ref(),
        &matrix,
        k,
    );
    run(
        "squarederror_k3_1m_outputs",
        objective(base().objective("reg:squarederror"), k).as_ref(),
        &matrix,
        k,
    );
    let third = meta_rows(N / k)
        .with_labels(&regression_labels(N / k))
        .unwrap();
    let quantile = objective(
        base()
            .objective("reg:quantileerror")
            .quantile_alpha(vec![0.1, 0.5, 0.9]),
        1,
    );
    run("quantile_a3_1m_outputs", quantile.as_ref(), &third, k);
    let expectile = objective(
        base()
            .objective("reg:expectileerror")
            .expectile_alpha(vec![0.1, 0.5, 0.9]),
        1,
    );
    run("expectile_a3_1m_outputs", expectile.as_ref(), &third, k);
    let lower: Vec<f32> = (0..N).map(|i| 0.5 + (i % 101) as f32 * 0.03).collect();
    let upper: Vec<f32> = lower
        .iter()
        .enumerate()
        .map(|(i, &l)| match i % 3 {
            0 => l,
            1 => l * 1.5,
            _ => f32::INFINITY,
        })
        .collect();
    let bounds = meta_rows(N).with_label_bounds(&lower, &upper).unwrap();
    run(
        "aft_normal_1m",
        objective(base().objective("survival:aft"), 1).as_ref(),
        &bounds,
        1,
    );
    let n = 100_000;
    let times: Vec<f32> = (0..n)
        .map(|i| if i % 4 == 0 { -1.0 } else { 1.0 } * (1.0 + (i * 37 % 1_000) as f32 * 0.01))
        .collect();
    run(
        "cox_100k",
        objective(base().objective("survival:cox"), 1).as_ref(),
        &meta_rows(n).with_labels(&times).unwrap(),
        1,
    );
    let ranked = meta_rows(n)
        .with_labels(&relevance_labels(n))
        .unwrap()
        .with_group_sizes(&group_sizes(n, 100))
        .unwrap();
    for name in ["rank:ndcg", "rank:map", "rank:pairwise"] {
        let id = format!("{}_100k_groups100", name.replace(':', "_"));
        run(
            &id,
            objective(base().objective(name), 1).as_ref(),
            &ranked,
            1,
        );
    }
    group.finish();
}

/// Evaluation metrics outside the pointwise kernels, through `eval_info`
/// (the training path): curves, ranking, survival, alpha lists, the
/// elementwise metrics, and the `dist:*` scores.
fn bench_other_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("eval_metric_other");
    let n = 100_000;
    let mut run = |name: &str, params: &TrainingParams, data: &DMatrix, preds: &[f32]| {
        let metric = create_metric(name.split('/').next().unwrap(), params).unwrap();
        let info = data.info();
        group.throughput(Throughput::Elements(data.n_rows() as u64));
        group.bench_function(name.replace('/', "_"), |b| {
            b.iter(|| black_box(metric.eval_info(preds, &info)));
        });
    };
    let default = TrainingParams::default();
    let scores: Vec<f32> = (0..3 * n)
        .map(|i| ((i * 7_919) % 10_007) as f32 / 10_007.0)
        .collect();
    let binary = meta_rows(n).with_labels(&alternating_labels(n)).unwrap();
    let weighted = meta_rows(n)
        .with_labels(&alternating_labels(n))
        .unwrap()
        .with_weights(&make_weights(n))
        .unwrap();
    run("auc/100k", &default, &binary, &scores[..n]);
    run("auc/100k_weighted", &default, &weighted, &scores[..n]);
    run("aucpr/100k", &default, &binary, &scores[..n]);
    let matrix = meta_rows(n)
        .with_label_matrix(&alternating_labels(3 * n), 3)
        .unwrap();
    run("auc/100k_k3_matrix", &default, &matrix, &scores);
    let ranked = meta_rows(n)
        .with_labels(&relevance_labels(n))
        .unwrap()
        .with_group_sizes(&group_sizes(n, 100))
        .unwrap();
    for name in ["ndcg", "ndcg@10", "map", "map@10", "pre@5", "auc"] {
        run(
            &format!("{name}/100k_groups100"),
            &default,
            &ranked,
            &scores[..n],
        );
    }
    let positive = meta_rows(N).with_labels(&positive_labels(N)).unwrap();
    let predictions: Vec<f32> = (0..N).map(|i| 0.3 + (i % 997) as f32 * 0.002).collect();
    for name in ["rmsle", "mape", "mphe"] {
        run(&format!("{name}/1m"), &default, &positive, &predictions);
    }
    let alphas = TrainingParams::builder()
        .objective("reg:quantileerror")
        .quantile_alpha(vec![0.1, 0.5, 0.9])
        .build()
        .unwrap();
    let third = meta_rows(n).with_labels(&positive_labels(n)).unwrap();
    run("quantile/100k_a3", &alphas, &third, &scores);
    let lower: Vec<f32> = (0..n).map(|i| 0.5 + (i % 101) as f32 * 0.03).collect();
    let upper: Vec<f32> = lower
        .iter()
        .enumerate()
        .map(|(i, &l)| {
            if i % 3 == 2 {
                f32::INFINITY
            } else {
                l * (1.0 + (i % 3) as f32)
            }
        })
        .collect();
    let bounds = meta_rows(n).with_label_bounds(&lower, &upper).unwrap();
    let aft = TrainingParams::builder()
        .objective("survival:aft")
        .build()
        .unwrap();
    run("aft-nloglik/100k", &aft, &bounds, &predictions[..n]);
    run(
        "interval-regression-accuracy/100k",
        &aft,
        &bounds,
        &predictions[..n],
    );
    let times: Vec<f32> = (0..n)
        .map(|i| if i % 4 == 0 { -1.0 } else { 1.0 } * (1.0 + (i * 37 % 1_000) as f32 * 0.01))
        .collect();
    let cox = TrainingParams::builder()
        .objective("survival:cox")
        .build()
        .unwrap();
    run(
        "cox-nloglik/100k",
        &cox,
        &meta_rows(n).with_labels(&times).unwrap(),
        &predictions[..n],
    );
    // `dist:*` predictions: two natural parameters per row.
    let counts: Vec<f32> = (0..n).map(|i| (i % 13) as f32).collect();
    let params: Vec<f32> = (0..2 * n)
        .map(|i| {
            if i % 2 == 0 {
                0.5 + (i % 29) as f32 * 0.1
            } else {
                0.5 + (i % 7) as f32 * 0.2
            }
        })
        .collect();
    for (family, labels) in [
        ("dist:normal", &regression_like(n)),
        ("dist:gamma", &positive_labels(n)),
        ("dist:negbinomial", &counts),
    ] {
        let dist = TrainingParams::builder().objective(family).build().unwrap();
        let data = meta_rows(n).with_labels(labels).unwrap();
        for metric in ["nll", "crps"] {
            run(
                &format!("{metric}/100k_{}", family.trim_start_matches("dist:")),
                &dist,
                &data,
                &params,
            );
        }
    }
    group.finish();
}

/// Real-valued labels around zero for the `dist:normal` metric benches.
fn regression_like(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 997) as f32 * 0.004 - 2.0).collect()
}

/// End-to-end training of the configurations the core benches do not cover:
/// the opt-in growers, sampling and constraints, categorical splits, DART,
/// sparse input, `approx` forests, and per-round eval-set metrics.
fn bench_train_variants(c: &mut Criterion) {
    let data = make_data(50_000, 20);
    let categorical = make_categorical_data(50_000, 20);
    let sparse = make_csr_data(50_000, 20);
    let eval = make_data(20_000, 20);
    let matrix = {
        let labels = data.labels().unwrap();
        let y: Vec<f32> = labels
            .iter()
            .flat_map(|&y| [y, y * 0.5 - 1.0, 2.0 - y])
            .collect();
        make_data(50_000, 20).with_label_matrix(&y, 3).unwrap()
    };
    let mut group = c.benchmark_group("train_variants_50k_x20_20rounds");
    group.sample_size(10);
    let base = || {
        TrainingParams::builder()
            .objective("reg:squarederror")
            .tree_method(TreeMethod::Hist)
            .max_depth(6)
            .eta(0.1)
    };
    let cases: Vec<(&str, TrainingParamsBuilder, &DMatrix)> = vec![
        (
            "lossguide_colsample_bytree",
            base()
                .grow_policy(GrowPolicy::LossGuide)
                .max_leaves(64)
                .colsample_bytree(0.8),
            &data,
        ),
        (
            "colsample_bylevel_bynode",
            base().colsample_bylevel(0.7).colsample_bynode(0.7),
            &data,
        ),
        (
            "constraints",
            base()
                .monotone_constraints(vec![
                    Monotone::Increasing,
                    Monotone::None,
                    Monotone::Decreasing,
                ])
                .interaction_constraints(vec![
                    vec![0, 1, 2, 3, 4],
                    vec![4, 5, 6, 7, 8, 9],
                    (10..20).collect(),
                ]),
            &data,
        ),
        ("categorical", base(), &categorical),
        (
            "exact_categorical",
            base().tree_method(TreeMethod::Exact),
            &categorical,
        ),
        (
            "exact_colsample_bylevel",
            base().tree_method(TreeMethod::Exact).colsample_bylevel(0.6),
            &data,
        ),
        (
            "approx_forest4",
            base()
                .tree_method(TreeMethod::Approx)
                .num_parallel_tree(4)
                .subsample(0.8),
            &data,
        ),
        (
            "dart",
            base().booster(BoosterKind::Dart).rate_drop(0.1),
            &data,
        ),
        ("csr", base(), &sparse),
        (
            "multi_output_k3",
            base().multi_strategy(MultiStrategy::MultiOutputTree),
            &matrix,
        ),
        ("one_output_per_tree_k3", base(), &matrix),
        (
            "symmetric",
            base().grow_policy(GrowPolicy::Symmetric),
            &data,
        ),
        ("extra_trees", base().extra_trees(true), &data),
        ("path_smooth", base().path_smooth(1.0), &data),
        ("linear_tree", base().linear_tree(true), &data),
        (
            "reuse_penalties",
            base().toad_penalty_feature(0.5).toad_penalty_threshold(0.1),
            &data,
        ),
    ];
    for (name, params, dtrain) in cases {
        let params = params.build().unwrap();
        group.bench_function(name, |b| {
            b.iter(|| black_box(train(&params, dtrain, 20).unwrap()));
        });
    }
    let params = base()
        .eval_metric("rmse")
        .eval_metric("mae")
        .build()
        .unwrap();
    group.bench_function("eval_set_rmse_mae", |b| {
        b.iter(|| {
            black_box(
                Trainer::new(&params, &data, 20)
                    .eval(&eval, "eval")
                    .train()
                    .unwrap(),
            )
        });
    });
    let budget = TrainingParams::default();
    let config = BudgetConfig::new(1.0).iteration_limit(20);
    group.bench_function("budget", |b| {
        b.iter(|| black_box(train_with_budget(&budget, &data, &config).unwrap()));
    });
    group.finish();
}

/// Batch prediction from CSR input: the 100-tree model of
/// `predict_100k_x30_100trees_depth6` on a sparse copy of the rows, and a
/// model over 5,000 sparse columns (wider than the gathered block layout).
fn bench_predict_csr(c: &mut Criterion) {
    let mut group = c.benchmark_group("predict_csr_100trees_depth6");
    group.sample_size(20);
    let params = TrainingParams::builder()
        .tree_method(TreeMethod::Hist)
        .max_depth(6)
        .eta(0.1)
        .build()
        .unwrap();
    let model = train(&params, &make_data(100_000, 30), 100).unwrap();
    let sparse = make_csr_data(100_000, 30);
    group.throughput(Throughput::Elements(sparse.n_rows() as u64));
    group.bench_function("100k_x30", |b| {
        b.iter(|| model.predict_margin(black_box(&sparse)).unwrap());
    });
    let (n, f) = (20_000usize, 5_000usize);
    let mut indptr = vec![0usize];
    let (mut indices, mut values, mut labels) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let mut y = 0.0;
        for k in 0..10 {
            let j = (i * 131 + k * 997) % f;
            let v = ((i * 7 + k * 13) % 101) as f32 / 101.0;
            if j < 50 {
                y += v;
            }
            indices.push(j as u32);
            values.push(v);
        }
        // CSR rows need ascending, distinct columns.
        let row = indptr[i]..indices.len();
        let mut entries: Vec<(u32, f32)> = indices[row.clone()]
            .iter()
            .copied()
            .zip(values[row.clone()].iter().copied())
            .collect();
        entries.sort_by_key(|e| e.0);
        entries.dedup_by_key(|e| e.0);
        indices.truncate(row.start);
        values.truncate(row.start);
        for (j, v) in entries {
            indices.push(j);
            values.push(v);
        }
        indptr.push(indices.len());
        labels.push(y);
    }
    let wide = DMatrix::from_csr(indptr, indices, values, f)
        .unwrap()
        .with_labels(&labels)
        .unwrap();
    let model = train(&params, &wide, 100).unwrap();
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("20k_x5000_wide", |b| {
        b.iter(|| model.predict_margin(black_box(&wide)).unwrap());
    });
    group.finish();
}

/// Serializing and loading a 100-tree model in the native binary, native
/// JSON, and XGBoost JSON formats.
fn bench_model_io(c: &mut Criterion) {
    let params = TrainingParams::builder()
        .tree_method(TreeMethod::Hist)
        .max_depth(6)
        .eta(0.1)
        .build()
        .unwrap();
    let model = train(&params, &make_data(20_000, 20), 100).unwrap();
    let bytes = model.to_bytes().unwrap();
    let json = model.to_json().unwrap();
    let xgboost = model.to_xgboost_json().unwrap();
    let mut group = c.benchmark_group("model_io_100trees_depth6");
    group.sample_size(20);
    group.bench_function("to_bytes", |b| {
        b.iter(|| black_box(model.to_bytes().unwrap()));
    });
    group.bench_function("from_bytes", |b| {
        b.iter(|| black_box(BoostedModel::from_bytes(&bytes).unwrap()));
    });
    group.bench_function("to_json", |b| {
        b.iter(|| black_box(model.to_json().unwrap()));
    });
    group.bench_function("from_json", |b| {
        b.iter(|| black_box(BoostedModel::from_json(&json).unwrap()));
    });
    group.bench_function("to_xgboost_json", |b| {
        b.iter(|| black_box(model.to_xgboost_json().unwrap()));
    });
    group.bench_function("from_xgboost_json", |b| {
        b.iter(|| black_box(BoostedModel::from_xgboost_json(&xgboost).unwrap()));
    });
    group.finish();
}

/// Data preparation before the first round: quantile cuts and the binned
/// index, from dense and CSR input.
fn bench_data_prep(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_prep_100k_x30");
    group.sample_size(20);
    for (name, data) in [
        ("dense", make_data(100_000, 30)),
        ("csr", make_csr_data(100_000, 30)),
    ] {
        group.bench_function(format!("cuts_{name}"), |b| {
            b.iter(|| black_box(HistCuts::from_dmatrix(&data, 256)));
        });
        let cuts = HistCuts::from_dmatrix(&data, 256);
        group.bench_function(format!("ghist_{name}"), |b| {
            b.iter(|| black_box(GHistIndex::from_dmatrix(&data, cuts.clone())));
        });
    }
    group.finish();
}

/// Metal GPU benches (`cargo bench --features metal` on a Mac with a Metal
/// device): histogram construction, end-to-end training, and batch
/// prediction, each against its CPU counterpart on identical data. The GPU
/// results are bit-identical to the single-threaded CPU's, so the benches
/// compare speed only.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn bench_metal(c: &mut Criterion) {
    use hessboost::backend::metal;
    use hessboost::backend::metal::MetalHistBackend;
    use hessboost::config::Device;

    if let Some(reason) = metal::unavailable_reason() {
        eprintln!("skipping metal benches: {reason}");
        return;
    }
    // Histogram construction at the sizes where the GPU pays off.
    {
        let mut group = c.benchmark_group("metal_histogram_build");
        group.sample_size(10);
        for &n in &[100_000usize, 1_000_000] {
            let (ghist, gpair, rows) = histogram_case(&make_data(n, 30));
            let gpu = MetalHistBackend::new(&ghist).unwrap();
            let mut cpu_out = zeroed(ghist.total_bins());
            let mut gpu_out = zeroed(ghist.total_bins());
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new("cpu", n), &n, |b, _| {
                b.iter(|| CpuBackend.build(&ghist, &rows, &gpair, &mut cpu_out));
            });
            group.bench_with_input(BenchmarkId::new("metal", n), &n, |b, _| {
                b.iter(|| gpu.build(&ghist, &rows, &gpair, &mut gpu_out));
            });
        }
        group.finish();
    }
    // End-to-end training: identical parameters, CPU against GPU histograms.
    {
        let data = make_data(200_000, 30);
        let mut group = c.benchmark_group("metal_train_200k_x30_50rounds_depth8");
        group.sample_size(10);
        for (name, device) in [("cpu", Device::Cpu), ("metal", Device::Metal)] {
            let params = TrainingParams::builder()
                .objective("reg:squarederror")
                .tree_method(TreeMethod::Hist)
                .max_depth(8)
                .eta(0.1)
                .device(device)
                .build()
                .unwrap();
            group.bench_function(name, |b| {
                b.iter(|| black_box(train(&params, &data, 50).unwrap()));
            });
        }
        group.finish();
    }
    // Batch prediction: the compact-forest walk against the GPU walk.
    {
        let model_data = make_data(100_000, 30);
        let params = TrainingParams::builder()
            .objective("reg:squarederror")
            .tree_method(TreeMethod::Hist)
            .max_depth(6)
            .eta(0.1)
            .build()
            .unwrap();
        let model = train(&params, &model_data, 100).unwrap();
        let gpu = model.to_gpu().unwrap();
        let data = make_data(500_000, 30);
        let mut group = c.benchmark_group("metal_predict_500k_x30_100trees_depth6");
        group.sample_size(10);
        group.throughput(Throughput::Elements(data.n_rows() as u64));
        group.bench_function("cpu", |b| {
            b.iter(|| model.predict_margin(black_box(&data)).unwrap());
        });
        group.bench_function("metal", |b| {
            b.iter(|| gpu.predict_margin(black_box(&data)).unwrap());
        });
        group.finish();
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn bench_metal_registered(c: &mut Criterion) {
    bench_metal(c);
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn bench_metal_registered(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_histogram_build,
    bench_hist_tree_build,
    bench_objective_gradients,
    bench_prediction_transforms,
    bench_pointwise_metrics,
    bench_log_metrics,
    bench_multiclass_metrics,
    bench_binary_train,
    bench_train,
    bench_predict,
    bench_shap,
    bench_other_gradients,
    bench_other_metrics,
    bench_train_variants,
    bench_predict_csr,
    bench_model_io,
    bench_data_prep,
    bench_metal_registered
);
criterion_main!(benches);
