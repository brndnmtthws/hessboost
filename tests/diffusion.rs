//! Conditional diffusion and flow matching (`hessboost::diffusion`): the
//! learned distribution's shape, sampling determinism, persistence, and
//! refusals.

use hessboost::diffusion::{
    DiffusionModel, DiffusionParams, FlowMatchingConfig, FlowPath, Method, ScoreConfig, Sde,
};
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
    base.n_repeats = 10;
    base.num_boost_round = 150;
    base
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
        let samples = model.sample(&probe, 400, 3).unwrap();
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
    let samples = model.sample(&probes(&[0.5]), 300, 5).unwrap();
    assert_eq!(samples.values().len(), 300 * 2);
    let aligned = samples
        .values()
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
            model.sample(&probe, 50, 11).unwrap()
        })
    };
    assert_eq!(run(1), run(4));
}

#[test]
fn draws_depend_only_on_their_row_and_sample_index() {
    let data = bimodal(300, 3);
    let model = DiffusionModel::fit(&quick(DiffusionParams::flow_matching()), &data).unwrap();
    let all = model.sample(&probes(&[0.1, 0.5, 0.9]), 30, 7).unwrap();
    // Fewer samples: a prefix of each row's draws.
    let fewer = model.sample(&probes(&[0.1, 0.5, 0.9]), 10, 7).unwrap();
    for row in 0..3 {
        assert_eq!(fewer.row(row).unwrap(), &all.row(row).unwrap()[..10]);
    }
    // Fewer rows: the same draws for the rows kept.
    let first = model.sample(&probes(&[0.1]), 30, 7).unwrap();
    assert_eq!(first.row(0), all.row(0));
    // Another seed: other draws.
    assert_ne!(model.sample(&probes(&[0.1]), 30, 8).unwrap(), first);
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
        let expected = model.sample(&probe, 20, 1).unwrap();
        let from_bytes = DiffusionModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
        let from_json = DiffusionModel::from_json(&model.to_json().unwrap()).unwrap();
        for loaded in [&from_bytes, &from_json] {
            assert_eq!(loaded.method(), model.method());
            assert_eq!(loaded.n_steps(), model.n_steps());
            assert_eq!(loaded.sample(&probe, 20, 1).unwrap(), expected);
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
}

#[test]
fn unsupported_inputs_are_refused() {
    let data = bimodal(300, 6);
    let mut params = quick(DiffusionParams::default());
    params.training.objective = "reg:absoluteerror".into();
    assert_eq!(
        invalid_param(DiffusionModel::fit(&params, &data)),
        "training"
    );

    let mut params = quick(DiffusionParams::default());
    params.n_repeats = 0;
    assert_eq!(
        invalid_param(DiffusionModel::fit(&params, &data)),
        "n_repeats"
    );

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
        invalid_param(model.sample(&probes(&[0.5]), 0, 1)),
        "n_samples"
    );
    assert!(matches!(
        model.sample(&DMatrix::from_dense(&[0.5, 0.5], 1, 2).unwrap(), 5, 1),
        Err(HessboostError::DimensionMismatch { .. })
    ));
}
