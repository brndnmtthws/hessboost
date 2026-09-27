//! Conditional diffusion and flow matching (`hessboost::diffusion`): the
//! learned distribution's shape, sampling determinism, persistence, and
//! refusals.

use std::num::NonZeroUsize;

use hessboost::config::{
    BalancedBagging, BoosterKind, Boulevard, Ebm, ProcessType, QueryBagging, Refresh,
};
use hessboost::diffusion::{
    DiffusionModel, DiffusionParams, FlowMatchingConfig, FlowPath, Method, SampleOptions, Samples,
    ScoreConfig, Sde,
};
use hessboost::objective::LambdaRank;
use hessboost::prelude::*;

mod common;
use common::{invalid_param, lcg, with_threads};

/// `y = ±(1 + x) + 0.05 ε`: two modes whose gap grows with `x`.
fn bimodal(n: usize, seed: u64) -> DMatrix {
    let mut next = lcg(seed);
    let (mut x, mut y) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let xi = next();
        let sign = if next() < 0.5 { -1.0 } else { 1.0 };
        let noise = next() + next() + next() - 1.5; // mean 0, sd 0.5
        x.push(xi);
        y.push(sign * (1.0 + xi) + 0.1 * noise);
    }
    DMatrix::from_dense(&x, n, 1)
        .unwrap()
        .with_labels(&y)
        .unwrap()
}

/// A small, fast configuration of `base`.
fn quick(mut base: DiffusionParams) -> DiffusionParams {
    base.n_repeats = nz(10);
    base.num_boost_round = nz(150);
    base
}

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

fn probes(xs: &[f32]) -> DMatrix {
    DMatrix::from_dense(xs, xs.len(), 1).unwrap()
}

#[test]
fn samples_recover_both_modes() {
    let data = bimodal(600, 1);
    let probe = probes(&[0.2, 0.8]);
    for params in [
        quick(DiffusionParams::default()),
        quick(DiffusionParams::treeffuser()),
        quick(DiffusionParams::flow_matching()),
    ] {
        let model = DiffusionModel::fit(&params, &data).unwrap();
        let samples = model
            .sample(&probe, 400, &SampleOptions::seeded(3))
            .unwrap();
        for (row, x) in [0.2f32, 0.8].into_iter().enumerate() {
            let draws = samples.row(row).unwrap();
            let mode = 1.0 + x;
            // Most draws sit near one of the two modes, each mode holds a
            // good share, and few fall in the empty middle (a unimodal fit
            // would put its mass there). The residualized recipes blur and
            // unbalance sharp modes somewhat (as DiffGBM's reference code
            // does), hence the loose bands.
            let near = |m: f32| draws.iter().filter(|&&v| (v - m).abs() < 0.5).count();
            let (upper, lower) = (near(mode), near(-mode));
            let middle = draws.iter().filter(|v| v.abs() < 0.5).count();
            assert!(
                upper + lower > 280,
                "{:?}: {upper} + {lower}",
                params.method
            );
            assert!(
                upper > 90 && lower > 90,
                "{:?}: {upper} vs {lower}",
                params.method
            );
            assert!(middle < 40, "{:?}: {middle} in the gap", params.method);
        }
    }
}

#[test]
fn multivariate_labels_are_sampled_jointly() {
    // y = (u, -u) with u ~ ±1: the columns are perfectly anti-correlated,
    // which independent per-column marginals cannot reproduce.
    let n = 400;
    let mut next = lcg(9);
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let u = if next() < 0.5 { -1.0 } else { 1.0 };
        x.push(next());
        y.extend_from_slice(&[u, -u]);
    }
    let data = DMatrix::from_dense(&x, n, 1)
        .unwrap()
        .with_label_matrix(&y, 2)
        .unwrap();
    let model = DiffusionModel::fit(&quick(DiffusionParams::default()), &data).unwrap();
    assert_eq!(model.n_outputs(), 2);
    let samples = model
        .sample(&probes(&[0.5]), 300, &SampleOptions::seeded(5))
        .unwrap();
    assert_eq!(samples.as_slice().len(), 300 * 2);
    let aligned = samples
        .as_slice()
        .as_chunks::<2>()
        .0
        .iter()
        .filter(|p| (p[0] + p[1]).abs() < 0.5 && p[0].abs() > 0.5)
        .count();
    assert!(aligned > 250, "{aligned} of 300 draws keep y2 = -y1");
}

#[test]
fn fitting_and_sampling_ignore_the_thread_count() {
    let data = bimodal(300, 2);
    let probe = probes(&[0.1, 0.5, 0.9]);
    let run = |threads| {
        with_threads(threads, || {
            let model = DiffusionModel::fit(&quick(DiffusionParams::default()), &data).unwrap();
            model
                .sample(&probe, 50, &SampleOptions::seeded(11))
                .unwrap()
        })
    };
    assert_eq!(run(1), run(4));
}

/// A per-call step count samples exactly as a model trained to take that
/// many steps by default; without one, the model's own count applies.
#[test]
fn per_call_step_counts_override_the_stored_default() {
    let data = bimodal(200, 2);
    let probe = probes(&[0.3, 0.7]);
    let mut three = quick(DiffusionParams::flow_matching());
    three.n_steps = nz(3);
    let stored = DiffusionModel::fit(&three, &data).unwrap();
    let model = DiffusionModel::fit(&quick(DiffusionParams::flow_matching()), &data).unwrap();
    assert_eq!(model.n_steps(), nz(5));
    let options = SampleOptions::seeded(4);
    let overridden = model
        .sample(&probe, 20, &options.with_n_steps(nz(3)))
        .unwrap();
    assert_eq!(overridden, stored.sample(&probe, 20, &options).unwrap());
    assert_ne!(overridden, model.sample(&probe, 20, &options).unwrap());
}

#[test]
fn draws_depend_only_on_their_row_and_sample_index() {
    let data = bimodal(300, 3);
    let model = DiffusionModel::fit(&quick(DiffusionParams::flow_matching()), &data).unwrap();
    let all = model
        .sample(&probes(&[0.1, 0.5, 0.9]), 30, &SampleOptions::seeded(7))
        .unwrap();
    // Fewer samples: a prefix of each row's draws.
    let fewer = model
        .sample(&probes(&[0.1, 0.5, 0.9]), 10, &SampleOptions::seeded(7))
        .unwrap();
    for row in 0..3 {
        assert_eq!(fewer.row(row).unwrap(), &all.row(row).unwrap()[..10]);
    }
    // Fewer rows: the same draws for the rows kept.
    let first = model
        .sample(&probes(&[0.1]), 30, &SampleOptions::seeded(7))
        .unwrap();
    assert_eq!(first.row(0), all.row(0));
    // Another seed: other draws.
    assert_ne!(
        model
            .sample(&probes(&[0.1]), 30, &SampleOptions::seeded(8))
            .unwrap(),
        first
    );
}

#[test]
fn stored_draws_rebuild_their_samples() {
    let data = bimodal(300, 5);
    let model = DiffusionModel::fit(&quick(DiffusionParams::flow_matching()), &data).unwrap();
    let samples = model
        .sample(&probes(&[0.2, 0.6, 0.9]), 40, &SampleOptions::seeded(1))
        .unwrap();
    let rebuilt = Samples::new(samples.as_slice().to_vec(), 40, 1).unwrap();
    assert_eq!(rebuilt, samples);
    assert_eq!(rebuilt.n_rows(), 3);
    // Wrong layouts and non-finite draws are refused.
    for (values, n_samples, n_outputs) in [
        (vec![0.0; 6], 4, 1),
        (vec![0.0; 6], 0, 1),
        (vec![0.0; 6], 3, 0),
        (vec![0.0, f32::NAN], 2, 1),
        (Vec::new(), usize::MAX, 2),
    ] {
        assert_eq!(
            invalid_param(Samples::new(values, n_samples, n_outputs)),
            "samples"
        );
    }
}

#[test]
fn summaries_of_no_rows_are_empty() {
    // Without rows, `n_samples` is not bounded by the draws' length: the
    // summaries must not size buffers by it (2⁴⁰ `f64`s cannot be allocated).
    let empty = Samples::new(Vec::new(), 1 << 40, 1).unwrap();
    assert_eq!(empty.n_rows(), 0);
    assert!(empty.mean().as_slice().is_empty());
    let q = empty.quantiles(&[0.1, 0.9]).unwrap();
    assert_eq!((q.n_rows(), q.n_levels(), q.n_outputs()), (0, 2, 1));
    assert!(q.as_slice().is_empty());
    assert!(empty.crps(&[]).unwrap().as_slice().is_empty());
}

#[test]
fn both_formats_round_trip_the_sampler() {
    let data = bimodal(300, 4);
    let probe = probes(&[0.3, 0.7]);
    let mut treeffuser = quick(DiffusionParams::treeffuser());
    let mut vp = ScoreConfig::treeffuser();
    vp.sde = Sde::VariancePreserving {
        beta_min: 0.1,
        beta_max: 20.0,
    };
    treeffuser.method = Method::Score(vp);
    for params in [
        quick(DiffusionParams::default()),
        treeffuser,
        quick(DiffusionParams::flow_matching()),
    ] {
        let model = DiffusionModel::fit(&params, &data).unwrap();
        let expected = model.sample(&probe, 20, &SampleOptions::seeded(1)).unwrap();
        let bytes = model.to_bytes().unwrap();
        let from_bytes = DiffusionModel::from_bytes(&bytes).unwrap();
        assert!(
            from_bytes.to_bytes().unwrap() == bytes,
            "re-saving changes the bytes"
        );
        let from_json = DiffusionModel::from_json(&model.to_json().unwrap()).unwrap();
        for loaded in [&from_bytes, &from_json] {
            assert_eq!(loaded.method(), model.method());
            assert_eq!(loaded.n_steps(), model.n_steps());
            assert_eq!(
                loaded
                    .sample(&probe, 20, &SampleOptions::seeded(1))
                    .unwrap(),
                expected
            );
        }
    }
}

#[test]
fn damaged_or_incomplete_files_are_refused() {
    let data = bimodal(300, 5);
    let model = DiffusionModel::fit(&quick(DiffusionParams::default()), &data).unwrap();
    let bytes = model.to_bytes().unwrap();
    for truncated in [&bytes[..bytes.len() / 2], &bytes[..3], &[][..]] {
        assert!(matches!(
            DiffusionModel::from_bytes(truncated),
            Err(HessboostError::ModelFormat(_))
        ));
    }
    // A native GBDT file is not a diffusion model, and vice versa.
    let gbdt = model.regressor().to_bytes().unwrap();
    assert!(matches!(
        DiffusionModel::from_bytes(&gbdt),
        Err(HessboostError::ModelFormat(_))
    ));
    assert!(BoostedModel::from_bytes(&bytes).is_err());

    let mut json: serde_json::Value = serde_json::from_str(&model.to_json().unwrap()).unwrap();
    json.as_object_mut().unwrap().remove("residualizer");
    assert!(DiffusionModel::from_json(&json.to_string()).is_err());
    let mut json: serde_json::Value = serde_json::from_str(&model.to_json().unwrap()).unwrap();
    json["target_scale"] = serde_json::json!([0.0]);
    assert!(matches!(
        DiffusionModel::from_json(&json.to_string()),
        Err(HessboostError::Json(_) | HessboostError::ModelFormat(_))
    ));
    // A regressor weighted by `scale_pos_weight` is not the unweighted
    // squared-error model the sampler integrates.
    let mut json: serde_json::Value = serde_json::from_str(&model.to_json().unwrap()).unwrap();
    json["regressor"]["objective_params"]["scale_pos_weight"] = serde_json::json!(2.0);
    let refused = DiffusionModel::from_json(&json.to_string()).unwrap_err();
    assert!(
        refused.to_string().contains("scale_pos_weight 1"),
        "{refused}"
    );
}

#[test]
fn refresh_regressor_params_are_refused() {
    // Every GBDT is trained from scratch: there is nothing to refresh.
    let update = ProcessType::Update(Refresh::default());
    let mut params = quick(DiffusionParams::default());
    params.training.process_type = update;
    assert_eq!(invalid_param(params.validate()), "training");
    assert_eq!(
        invalid_param(DiffusionModel::fit(&params, &bimodal(100, 7))),
        "training"
    );
    let mut params = quick(DiffusionParams::default());
    params.residualizer.as_mut().unwrap().training.process_type = update;
    assert_eq!(invalid_param(params.validate()), "residualizer.training");
}

#[test]
fn row_bagging_regressor_params_are_refused() {
    // LightGBM's class-balanced and query-level bagging sample rows by class
    // or query; the score regressors fit `reg:squarederror` on the noisy
    // training set, which has neither. Both are refused alone and with the
    // objective each one needs, for the score GBDT and the residualizer.
    let balanced = || Some(BalancedBagging::new(0.5, 0.5).unwrap());
    let query = || Some(QueryBagging::new(0.5).unwrap());
    let configs: [&dyn Fn(&mut TrainingParams); 4] = [
        &|t| t.balanced_bagging = balanced(),
        &|t| t.bagging_by_query = query(),
        &|t| {
            t.objective = Objective::BinaryLogistic(RegLoss::default());
            t.balanced_bagging = balanced();
        },
        &|t| {
            t.objective = Objective::RankPairwise(LambdaRank::default());
            t.bagging_by_query = query();
        },
    ];
    let data = bimodal(100, 7);
    let refused = |params: &DiffusionParams| {
        assert!(matches!(
            params.validate(),
            Err(HessboostError::InvalidParameter { .. })
        ));
        assert!(matches!(
            DiffusionModel::fit(params, &data),
            Err(HessboostError::InvalidParameter { .. })
        ));
    };
    for set in configs {
        let mut params = quick(DiffusionParams::default());
        set(&mut params.training);
        refused(&params);
        let mut params = quick(DiffusionParams::default());
        set(&mut params.residualizer.as_mut().unwrap().training);
        refused(&params);
    }
}

#[test]
fn single_label_boosters_train_on_one_label_column() {
    // `booster = boulevard` (one averaged squared-error regressor) and
    // `booster = ebm` (additive terms) score a scalar target without early
    // stopping; their own refusals (early stopping, a label matrix) come
    // back as errors from `fit`, not panics.
    let boosters = [
        BoosterKind::Boulevard(Boulevard::default()),
        BoosterKind::Ebm(Ebm::default()),
    ];
    let data = bimodal(200, 4);
    let probe = probes(&[0.5]);
    let y: Vec<f32> = (0..200).flat_map(|i| [i as f32, -(i as f32)]).collect();
    let x: Vec<f32> = (0..200).map(|i| i as f32 / 200.0).collect();
    let wide = DMatrix::from_dense(&x, 200, 1)
        .unwrap()
        .with_label_matrix(&y, 2)
        .unwrap();
    for booster in boosters {
        let mut params = quick(DiffusionParams::treeffuser());
        params.training = TrainingParams::builder()
            .booster(booster)
            .eta(0.8)
            .build()
            .unwrap();
        assert_eq!(
            invalid_param(DiffusionModel::fit(&params, &data)),
            "early_stopping_rounds"
        );
        params.early_stopping = None;
        let model = DiffusionModel::fit(&params, &data).unwrap();
        let samples = model.sample(&probe, 50, &SampleOptions::seeded(1)).unwrap();
        let reloaded = DiffusionModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
        assert_eq!(
            reloaded
                .sample(&probe, 50, &SampleOptions::seeded(1))
                .unwrap(),
            samples
        );
        assert!(matches!(
            DiffusionModel::fit(&params, &wide),
            Err(HessboostError::InvalidParameter { .. })
        ));
    }
}

#[test]
fn unsupported_inputs_are_refused() {
    let data = bimodal(300, 6);
    let mut params = quick(DiffusionParams::default());
    params.training.objective = Objective::AbsoluteError;
    assert_eq!(
        invalid_param(DiffusionModel::fit(&params, &data)),
        "training"
    );
    // `scale_pos_weight` would reweight the rows whose noisy target is
    // positive: only the unweighted squared error is accepted, for the
    // score GBDT and the residualizer.
    let weighted = || Objective::SquaredError(RegLoss::new(2.0).unwrap());
    let mut params = quick(DiffusionParams::default());
    params.training.objective = weighted();
    assert_eq!(invalid_param(params.validate()), "training");
    let mut params = quick(DiffusionParams::default());
    params.residualizer.as_mut().unwrap().training.objective = weighted();
    assert_eq!(
        invalid_param(DiffusionModel::fit(&params, &data)),
        "residualizer.training"
    );
    let mut params = quick(DiffusionParams::default());
    params.training.objective = Objective::SquaredError(RegLoss::new(1.0).unwrap());
    assert!(params.validate().is_ok());

    let mut params = quick(DiffusionParams::default());
    let mut broken = ScoreConfig::default();
    broken.sde = Sde::VarianceExploding {
        sigma_min: 1.0,
        sigma_max: f64::NAN,
    };
    params.method = Method::Score(broken);
    assert_eq!(invalid_param(DiffusionModel::fit(&params, &data)), "sde");

    let weighted = bimodal(300, 6).with_weights(&[1.0; 300]).unwrap();

    // Finite bounds whose kernel overflows (σ_max² > f64::MAX) are refused,
    // not a panic in the time sampling; so is a schedule whose noise scale
    // vanishes, and likewise for a flow path.
    for (sigma_min, sigma_max) in [(1e200, 2e200), (1e-320, 1e-310)] {
        let mut params = quick(DiffusionParams::default());
        params.residualizer = None;
        params.early_stopping = None;
        let mut overflowing = ScoreConfig::default();
        overflowing.sde = Sde::VarianceExploding {
            sigma_min,
            sigma_max,
        };
        params.method = Method::Score(overflowing);
        assert_eq!(invalid_param(DiffusionModel::fit(&params, &data)), "sde");
    }
    let mut params = quick(DiffusionParams::flow_matching());
    params.residualizer = None;
    params.early_stopping = None;
    let mut vanishing = FlowMatchingConfig::default();
    vanishing.path = FlowPath::VariancePreserving {
        beta_min: 1e-321,
        beta_max: 2e-321,
    };
    params.method = Method::FlowMatching(vanishing);
    assert_eq!(invalid_param(DiffusionModel::fit(&params, &data)), "path");
    assert_eq!(
        invalid_param(DiffusionModel::fit(
            &quick(DiffusionParams::default()),
            &weighted
        )),
        "data"
    );
    let unlabelled = DMatrix::from_dense(&[0.0; 10], 10, 1).unwrap();
    assert_eq!(
        invalid_param(DiffusionModel::fit(
            &quick(DiffusionParams::treeffuser()),
            &unlabelled
        )),
        "data"
    );
    // Residualization cross-fits on at least 80 rows.
    let small = bimodal(50, 6);
    assert_eq!(
        invalid_param(DiffusionModel::fit(
            &quick(DiffusionParams::default()),
            &small
        )),
        "residualizer"
    );
    let model = DiffusionModel::fit(&quick(DiffusionParams::treeffuser()), &small).unwrap();
    assert_eq!(
        invalid_param(model.sample(&probes(&[0.5]), 0, &SampleOptions::seeded(1))),
        "n_samples"
    );
    // Requests too large to allocate are refused, not a capacity-overflow
    // panic: the samples, and the noisy training set (whose element count
    // fits in `usize` but whose bytes exceed `isize::MAX`).
    assert_eq!(
        invalid_param(model.sample(&probes(&[0.5]), usize::MAX, &SampleOptions::seeded(1))),
        "n_samples"
    );
    assert_eq!(
        invalid_param(model.sample(&probes(&[0.5]), usize::MAX / 2, &SampleOptions::seeded(1))),
        "n_samples"
    );
    for n_repeats in [usize::MAX, (1usize << 62) / (50 * 4)] {
        let mut params = quick(DiffusionParams::default());
        params.residualizer = None;
        params.early_stopping = None;
        params.n_repeats = nz(n_repeats);
        assert_eq!(
            invalid_param(DiffusionModel::fit(&params, &small)),
            "n_repeats"
        );
    }
    assert!(matches!(
        model.sample(
            &DMatrix::from_dense(&[0.5, 0.5], 1, 2).unwrap(),
            5,
            &SampleOptions::seeded(1)
        ),
        Err(HessboostError::DimensionMismatch { .. })
    ));
}
