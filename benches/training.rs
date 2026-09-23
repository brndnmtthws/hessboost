//! Criterion benchmarks for histogram construction, objective and metric
//! kernels, prediction transforms, and end-to-end training.

use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use hessboost::config::ObjectiveParams;
use hessboost::data::ghist::GHistIndex;
use hessboost::data::quantile::HistCuts;
use hessboost::metric::{
    ErrorRate, GammaNLogLik, LogLoss, Mae, PoissonNLogLik, Rmse, create_metric,
};
use hessboost::objective::{
    GammaObjective, GradPair, LogisticObjective, Objective, PoissonObjective, SoftmaxObjective,
    TweedieObjective,
};
use hessboost::prelude::*;
use hessboost::tree::builder::HistTreeBuilder;
use hessboost::tree::hist::{CpuBackend, HistogramBackend, zeroed};
use hessboost::tree::sampler::ColumnSampler;
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
        let data = make_data(n, 30);
        let cuts = HistCuts::from_dmatrix(&data, 256);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        let gpair: Vec<GradPair> = (0..n)
            .map(|i| GradPair::new((i % 7) as f32 - 3.0, 1.0))
            .collect();
        let rows: Vec<u32> = (0..n as u32).collect();
        let mut out = zeroed(ghist.total_bins());

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| CpuBackend.build(&ghist, &rows, &gpair, &mut out));
        });
    }
    group.finish();
}

fn bench_hist_tree_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("hist_tree_build");
    for (name, n, features, depth, policy, quantized) in [
        ("depth1", 50_000, 20, 1, GrowPolicy::DepthWise, false),
        ("depth6", 50_000, 20, 6, GrowPolicy::DepthWise, false),
        ("depth10", 50_000, 20, 10, GrowPolicy::DepthWise, false),
        ("wide128", 10_000, 128, 6, GrowPolicy::DepthWise, false),
        ("missing", 50_000, 20, 6, GrowPolicy::DepthWise, false),
        ("monotone", 50_000, 20, 6, GrowPolicy::DepthWise, false),
        ("lossguide", 50_000, 20, 6, GrowPolicy::LossGuide, false),
        (
            "large_depth8",
            1_000_000,
            50,
            8,
            GrowPolicy::DepthWise,
            false,
        ),
        // Opt-in quantized gradients (`use_quantized_grad`); the rows above
        // with the same data are the full-precision baselines.
        (
            "depth6_quantized",
            50_000,
            20,
            6,
            GrowPolicy::DepthWise,
            true,
        ),
        (
            "depth10_quantized",
            50_000,
            20,
            10,
            GrowPolicy::DepthWise,
            true,
        ),
        (
            "wide128_quantized",
            10_000,
            128,
            6,
            GrowPolicy::DepthWise,
            true,
        ),
        (
            "missing_quantized",
            50_000,
            20,
            6,
            GrowPolicy::DepthWise,
            true,
        ),
        (
            "large_depth8_quantized",
            1_000_000,
            50,
            8,
            GrowPolicy::DepthWise,
            true,
        ),
    ] {
        let mut data = make_data(n, features);
        if name.starts_with("missing") {
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
        let params = TrainingParams::builder()
            .max_depth(depth)
            .grow_policy(policy)
            .max_leaves(64)
            .monotone_constraints(if name == "monotone" {
                vec![Monotone::Increasing]
            } else {
                Vec::new()
            })
            .use_quantized_grad(quantized)
            .build()
            .unwrap();
        let builder = HistTreeBuilder::new(&params);
        // A 1M-row tree takes long enough that ten samples are stable.
        group.sample_size(if n >= 1_000_000 { 10 } else { 100 });
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut sampler = ColumnSampler::all(features);
                black_box(builder.build(&ghist, &gpair, &rows, &mut sampler))
            });
        });
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
        &LogisticObjective::default(),
        &labels,
        None,
    );
    run(
        "logistic_weighted_1m",
        &LogisticObjective::new(1.5),
        &labels,
        Some(&weights),
    );
    run(
        "poisson_unweighted_1m",
        &PoissonObjective::default(),
        &positive_labels,
        None,
    );
    run(
        "gamma_unweighted_1m",
        &GammaObjective,
        &positive_labels,
        None,
    );
    run(
        "tweedie_unweighted_1m",
        &TweedieObjective::default(),
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
        let softmax = SoftmaxObjective::new(k, true);
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
            LogisticObjective::default().pred_transform(&mut values);
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
            GammaObjective.pred_transform(&mut values);
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
        let objective = SoftmaxObjective::new(num_class, true);
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
        ("rmse", &Rmse as &dyn Metric),
        ("mae", &Mae as &dyn Metric),
        ("error", &ErrorRate as &dyn Metric),
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

    for (name, metric, labels) in [
        ("logloss", &LogLoss as &dyn Metric, binary_labels.as_slice()),
        (
            "poisson_nloglik",
            &PoissonNLogLik as &dyn Metric,
            positive_labels.as_slice(),
        ),
        (
            "gamma_nloglik",
            &GammaNLogLik as &dyn Metric,
            positive_labels.as_slice(),
        ),
    ] {
        bench_metric_pair(&mut group, name, metric, &probabilities, labels, &weights);
    }
    let tweedie = create_metric("tweedie-nloglik@1.5", 0, &ObjectiveParams::default()).unwrap();
    bench_metric_pair(
        &mut group,
        "tweedie_nloglik",
        tweedie.as_ref(),
        &probabilities,
        &positive_labels,
        &weights,
    );
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
        for metric_name in ["mlogloss", "merror"] {
            let metric =
                create_metric(metric_name, num_class, &ObjectiveParams::default()).unwrap();
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
                train_with_objective(&params, &data, 50, &scalar_logistic_objective(base_margin))
                    .unwrap(),
            )
        });
    });
    let quantized = TrainingParams {
        use_quantized_grad: true,
        ..params.clone()
    };
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
    bench_shap
);
criterion_main!(benches);
