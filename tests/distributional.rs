//! Distributional boosting (`dist:*` objectives): training,
//! calibration of the predicted distributions, metrics, serialization, and
//! conformalized intervals.

use hessboost::config::{MaxDeltaStep, MultiStrategy, TrainingParamsBuilder, TreeMethod};
use hessboost::conformal::ConformalizedQuantile;
use hessboost::metric::EvalMetric;
use hessboost::objective::distributional::{
    Dist, DistFamily, DistGradient, DistSplitDirection, Distributional,
};
use hessboost::prelude::{
    BoostedModel, DMatrix, HessboostError, Objective, RegLoss, Trainer, TrainingParams, train,
};
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
use std::num::NonZeroUsize;

mod common;
use common::labeled_dense;

/// Heteroscedastic regression: `y = 2 sin(2π x₀) + (0.1 + x₁) ε`,
/// `ε ~ N(0, 1)`, with a third, irrelevant feature. Returns the matrix and
/// the true noise scale of every row.
fn heteroscedastic(n: usize, seed: u64) -> (DMatrix, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Dist::Normal {
        mu: 0.0,
        sigma: 1.0,
    };
    let mut x = Vec::with_capacity(3 * n);
    let mut y = Vec::with_capacity(n);
    let mut sigma = Vec::with_capacity(n);
    for _ in 0..n {
        let f: [f32; 3] = [rng.random(), rng.random(), rng.random()];
        let s = 0.1 + f64::from(f[1]);
        let mean = 2.0 * (std::f64::consts::TAU * f64::from(f[0])).sin();
        y.push((mean + s * normal.sample(|| rng.next_u64())) as f32);
        sigma.push(s);
        x.extend(f);
    }
    (labeled_dense(&x, 3, &y), sigma)
}

/// Rows drawn from `dist_of(x)` for two uniform features.
fn sampled(n: usize, seed: u64, dist_of: impl Fn(f64, f64) -> Dist) -> DMatrix {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Vec::with_capacity(2 * n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let f: [f32; 2] = [rng.random(), rng.random()];
        y.push(dist_of(f64::from(f[0]), f64::from(f[1])).sample(|| rng.next_u64()) as f32);
        x.extend(f);
    }
    labeled_dense(&x, 2, &y)
}

/// The `dist:*` objective of `family` with its default settings.
fn dist(family: DistFamily) -> Objective {
    Objective::Dist(Distributional::new(family))
}

fn params(objective: Objective) -> TrainingParamsBuilder {
    TrainingParams::builder()
        .objective(objective)
        .tree_method(TreeMethod::Hist)
        .max_depth(3)
        .eta(0.1)
        .seed(3)
}

/// Train with early stopping on `dvalid` (distributional models overfit
/// their scale parameters like any boosted model; the validation NLL picks
/// the iteration). Predictions then use the best iteration.
fn fit(p: &TrainingParams, dtrain: &DMatrix, dvalid: &DMatrix) -> BoostedModel {
    let result = Trainer::new(p, dtrain, 1000)
        .eval(dvalid, "valid")
        .early_stopping_rounds(NonZeroUsize::new(20).unwrap())
        .train()
        .unwrap();
    assert!(
        result.model.best_iteration().is_some(),
        "early stopping triggered"
    );
    result.model
}

fn mean_nll(dists: &[Dist], data: &DMatrix) -> f64 {
    let labels = data.labels().unwrap();
    dists
        .iter()
        .zip(labels)
        .map(|(d, &y)| -d.log_prob(f64::from(y)))
        .sum::<f64>()
        / labels.len() as f64
}

fn coverage(intervals: impl Iterator<Item = (f64, f64)>, labels: &[f32]) -> f64 {
    let hits = intervals
        .zip(labels)
        .filter(|&((lo, hi), &y)| lo <= f64::from(y) && f64::from(y) <= hi)
        .count();
    hits as f64 / labels.len() as f64
}

#[test]
fn heteroscedastic_intervals_are_calibrated_and_track_the_noise() {
    let (dtrain, _) = heteroscedastic(6000, 1);
    let (dvalid, _) = heteroscedastic(2000, 21);
    let (dtest, sigma) = heteroscedastic(6000, 2);
    let model = fit(
        &params(dist(DistFamily::Normal)).build().unwrap(),
        &dtrain,
        &dvalid,
    );
    assert_eq!(model.n_outputs(), 2);
    let dists = model.predict_distribution(&dtest).unwrap();
    let labels = dtest.labels().unwrap();
    for nominal in [0.5, 0.8, 0.95] {
        let got = coverage(dists.iter().map(|d| d.interval(nominal)), labels);
        // Binomial standard error at n = 6000 is below 0.007.
        assert!(
            (got - nominal).abs() < 0.03,
            "coverage {got} of nominal {nominal}"
        );
    }
    // The predicted scale follows the true noise scale.
    let rel_err = dists
        .iter()
        .zip(&sigma)
        .map(|(d, &s)| (d.std_dev() - s).abs() / s)
        .sum::<f64>()
        / sigma.len() as f64;
    assert!(rel_err < 0.2, "mean relative error of sigma {rel_err}");
    // `predict` reports the natural parameters (mu, sigma).
    let natural = model.predict(&dtest).unwrap();
    for (row, d) in natural.rows().zip(&dists) {
        let Dist::Normal { mu, sigma } = *d else {
            panic!("not a normal distribution")
        };
        assert!((f64::from(row[0]) - mu).abs() < 1e-5 * mu.abs().max(1.0));
        assert!((f64::from(row[1]) - sigma).abs() < 1e-5 * sigma);
    }
}

#[test]
fn distributional_nll_beats_a_homoscedastic_baseline() {
    let (dtrain, _) = heteroscedastic(4000, 3);
    let (dvalid, _) = heteroscedastic(2000, 31);
    let (dtest, _) = heteroscedastic(4000, 4);
    let dist = fit(
        &params(dist(DistFamily::Normal)).build().unwrap(),
        &dtrain,
        &dvalid,
    );
    let point = fit(
        &params(Objective::SquaredError(RegLoss::default()))
            .build()
            .unwrap(),
        &dtrain,
        &dvalid,
    );
    // Baseline: the point model with the training residuals' deviation.
    let fitted = point.predict(&dtrain).unwrap().into_vec(); // one value per row
    let train_labels = dtrain.labels().unwrap();
    let sd = (fitted
        .iter()
        .zip(train_labels)
        .map(|(&p, &y)| f64::from(y - p).powi(2))
        .sum::<f64>()
        / fitted.len() as f64)
        .sqrt();
    let baseline: Vec<Dist> = point
        .predict(&dtest)
        .unwrap()
        .as_slice()
        .iter()
        .map(|&mu| Dist::Normal {
            mu: f64::from(mu),
            sigma: sd,
        })
        .collect();
    let (nll_dist, nll_base) = (
        mean_nll(&dist.predict_distribution(&dtest).unwrap(), &dtest),
        mean_nll(&baseline, &dtest),
    );
    assert!(
        nll_dist < nll_base - 0.1,
        "dist {nll_dist} vs homoscedastic {nll_base}"
    );
}

/// Every family and gradient mode learns its covariate-dependent
/// parameters: the held-out NLL drops well below the intercept-only
/// (marginal MLE) fit, and the validation `nll` / `crps` history improves
/// and matches the predicted distributions.
#[test]
fn every_family_and_gradient_mode_learns() {
    type Truth = fn(f64, f64) -> Dist;
    let cases: [(DistFamily, Truth); 5] = [
        (DistFamily::Normal, |a, b| Dist::Normal {
            mu: 3.0 * a,
            sigma: 0.2 + b,
        }),
        (DistFamily::LogNormal, |a, b| Dist::LogNormal {
            mu: a,
            sigma: 0.2 + 0.8 * b,
        }),
        (DistFamily::Gamma, |a, b| Dist::Gamma {
            mean: 1.0 + 4.0 * a,
            shape: 0.5 + 6.0 * b,
        }),
        (DistFamily::Poisson, |a, _| Dist::Poisson {
            rate: 0.5 + 10.0 * a,
        }),
        (DistFamily::NegativeBinomial, |a, b| {
            Dist::NegativeBinomial {
                mean: 1.0 + 10.0 * a,
                size: 0.3 + 10.0 * b,
            }
        }),
    ];
    for (family, dist_of) in cases {
        let objective = family.objective_name();
        let dtrain = sampled(3000, 5, dist_of);
        let dvalid = sampled(2000, 6, dist_of);
        let dtest = sampled(3000, 7, dist_of);
        for mode in [
            DistGradient::Fisher,
            DistGradient::Hessian,
            DistGradient::Natural,
        ] {
            let p = params(Objective::Dist(
                Distributional::new(family).with_gradient(mode),
            ))
            .eval_metric(EvalMetric::Crps(family))
            .eval_metric(EvalMetric::Nll(family))
            .build()
            .unwrap();
            let result = Trainer::new(&p, &dtrain, 1000)
                .eval(&dvalid, "valid")
                .early_stopping_rounds(NonZeroUsize::new(20).unwrap())
                .train()
                .unwrap();
            let model = result.model;
            assert_eq!(model.n_outputs(), family.n_params());
            let best = model.best_iteration().expect("early stopping triggered");
            let first = result.history.round(0).unwrap().values();
            let at_best = result.history.round(best).unwrap().values();
            assert_eq!(result.history.metrics(), ["crps", "nll"]);
            assert!(
                at_best[0] < first[0] && at_best[1] < first[1],
                "{objective} {mode:?}"
            );
            // The reported metric is the mean NLL of the predicted
            // distributions (up to the f32 rounding of the parameters).
            let valid = mean_nll(&model.predict_distribution(&dvalid).unwrap(), &dvalid);
            assert!(
                (valid - at_best[1]).abs() < 1e-3 * valid.abs().max(1.0),
                "{objective}: {valid} vs {}",
                at_best[1]
            );
            let nll = mean_nll(&model.predict_distribution(&dtest).unwrap(), &dtest);
            let marginal = train(&p, &dtrain, 0).unwrap();
            let intercept_only = mean_nll(&marginal.predict_distribution(&dtest).unwrap(), &dtest);
            assert!(
                nll < intercept_only - 0.05,
                "{objective} {mode:?}: {nll} vs {intercept_only}"
            );
        }
    }
}

#[test]
fn dist_poisson_trains_like_count_poisson_without_max_delta_step() {
    let d = sampled(2000, 8, |a, b| Dist::Poisson {
        rate: 0.5 + 8.0 * a * b,
    });
    let fit = |objective: Objective| {
        let p = params(objective)
            .max_delta_step(MaxDeltaStep::Unbounded)
            .base_score(2.0)
            .build()
            .unwrap();
        train(&p, &d, 40).unwrap().predict(&d).unwrap()
    };
    let (dist, count) = (fit(dist(DistFamily::Poisson)), fit(Objective::Poisson));
    for (a, b) in dist.as_slice().iter().zip(count.as_slice()) {
        assert!((a - b).abs() <= 1e-4 * b.abs(), "{a} vs {b}");
    }
}

#[test]
fn native_round_trip_and_determinism() {
    let d = sampled(1500, 9, |a, b| Dist::Gamma {
        mean: 1.0 + a,
        shape: 1.0 + 5.0 * b,
    });
    let p = params(dist(DistFamily::Gamma))
        .subsample(0.8)
        .build()
        .unwrap();
    let model = train(&p, &d, 30).unwrap();
    let again = train(&p, &d, 30).unwrap();
    let margins = model.predict_margin(&d).unwrap();
    assert_eq!(
        margins,
        again.predict_margin(&d).unwrap(),
        "same seed, same model"
    );
    let dists = model.predict_distribution(&d).unwrap();
    for restored in [
        BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap(),
        BoostedModel::from_json(&model.to_json().unwrap()).unwrap(),
    ] {
        assert_eq!(restored.objective().name(), "dist:gamma");
        assert_eq!(restored.predict_distribution(&d).unwrap(), dists);
        assert_eq!(restored.predict(&d).unwrap(), model.predict(&d).unwrap());
    }
}

#[test]
fn xgboost_formats_refuse_distributional_models() {
    let d = sampled(300, 10, |a, _| Dist::Normal { mu: a, sigma: 1.0 });
    let model = train(&params(dist(DistFamily::Normal)).build().unwrap(), &d, 3).unwrap();
    assert!(matches!(
        model.to_xgboost_json(),
        Err(HessboostError::ModelFormat(_))
    ));
    assert!(matches!(
        model.to_xgboost_ubjson(),
        Err(HessboostError::ModelFormat(_))
    ));
    // A document claiming a `dist:*` objective is refused on import too.
    let point = train(
        &params(Objective::SquaredError(RegLoss::default()))
            .build()
            .unwrap(),
        &d,
        3,
    )
    .unwrap();
    let doc = point
        .to_xgboost_json()
        .unwrap()
        .replace("\"reg:squarederror\"", "\"dist:normal\"");
    assert!(matches!(
        BoostedModel::from_xgboost_json(&doc),
        Err(HessboostError::ModelFormat(_))
    ));
}

#[test]
fn conformalized_distribution_intervals_cover_misspecified_models() {
    // Heavy-tailed noise (a Student-t-like scale mixture) fitted with a
    // Normal: the raw 90% band under-covers, CQR restores the level.
    let make = |n: usize, seed: u64| {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let f: f32 = rng.random();
            let scale = if rng.random::<f64>() < 0.1 { 6.0 } else { 0.5 };
            let e = Dist::Normal {
                mu: 0.0,
                sigma: scale,
            }
            .sample(|| rng.next_u64());
            y.push((3.0 * f64::from(f) + e) as f32);
            x.push(f);
        }
        labeled_dense(&x, 1, &y)
    };
    let (dtrain, dcal, dtest) = (make(3000, 11), make(3000, 12), make(6000, 13));
    let model = fit(
        &params(dist(DistFamily::Normal)).build().unwrap(),
        &dtrain,
        &make(1000, 14),
    );
    let alpha = 0.1;
    let cqr = ConformalizedQuantile::calibrate_distribution(&model, &dcal, alpha).unwrap();
    assert_eq!(cqr.n_calibration(), 3000);
    let labels = dtest.labels().unwrap();
    let conformal = cqr.predict_interval(&dtest).unwrap();
    let got = coverage(
        conformal
            .iter()
            .map(|i| (f64::from(i.lower), f64::from(i.upper))),
        labels,
    );
    assert!(
        (got - (1.0 - alpha)).abs() < 0.02,
        "conformal coverage {got}"
    );
    // The raw band differs from the calibrated one by exactly the correction.
    let raw = model.predict_distribution(&dtest).unwrap();
    let (lo, hi) = raw[0].interval(1.0 - alpha);
    assert!((f64::from(conformal[0].lower) - (lo - cqr.correction())).abs() < 1e-4);
    assert!((f64::from(conformal[0].upper) - (hi + cqr.correction())).abs() < 1e-4);
    // Only distributional models qualify.
    let point = train(
        &params(Objective::SquaredError(RegLoss::default()))
            .build()
            .unwrap(),
        &dtrain,
        5,
    )
    .unwrap();
    assert!(ConformalizedQuantile::calibrate_distribution(&point, &dcal, alpha).is_err());
}

#[test]
fn configuration_errors() {
    let d = sampled(100, 14, |a, _| Dist::Normal { mu: a, sigma: 1.0 });
    // Labels outside the support.
    assert!(train(&params(dist(DistFamily::LogNormal)).build().unwrap(), &d, 1).is_err());
    assert!(train(&params(dist(DistFamily::Gamma)).build().unwrap(), &d, 1).is_err());
    // A scalar base_score cannot set two parameters; the Poisson rate can.
    let p = params(dist(DistFamily::Normal))
        .base_score(1.0)
        .build()
        .unwrap();
    assert!(train(&p, &d, 1).is_err());
    let counts = sampled(100, 15, |_, _| Dist::Poisson { rate: 2.0 });
    let p = params(dist(DistFamily::Poisson))
        .base_score(2.0)
        .build()
        .unwrap();
    let m = train(&p, &counts, 0).unwrap();
    assert!((m.base_score() - 2f32.ln()).abs() < 1e-7);
    // Label matrices are not distributional targets.
    let y = vec![0.5f32; 200];
    let multi = d.clone().with_label_matrix(&y, 2).unwrap();
    assert!(
        train(
            &params(dist(DistFamily::Normal)).build().unwrap(),
            &multi,
            1
        )
        .is_err()
    );
    // `nll` / `crps` score distributions: a point model's single output is
    // not a Normal's two parameters.
    let p = params(Objective::SquaredError(RegLoss::default()))
        .eval_metric(EvalMetric::Crps(DistFamily::Normal))
        .build()
        .unwrap();
    assert!(Trainer::new(&p, &d, 1).eval(&d, "d").train().is_err());
    // Point models do not predict distributions.
    let point = train(
        &params(Objective::SquaredError(RegLoss::default()))
            .build()
            .unwrap(),
        &d,
        1,
    )
    .unwrap();
    assert!(point.predict_distribution(&d).is_err());
}

/// Parallel gradient boosting (`multi_output_tree`): one shared tree per
/// round fits every distribution parameter, with the same quality as one
/// tree per parameter; round trips and is deterministic.
#[test]
fn shared_trees_fit_every_parameter_in_one_tree_per_round() {
    let (dtrain, _) = heteroscedastic(6000, 1);
    let (dvalid, _) = heteroscedastic(2000, 21);
    let (dtest, _) = heteroscedastic(6000, 2);
    let reference = fit(
        &params(dist(DistFamily::Normal)).build().unwrap(),
        &dtrain,
        &dvalid,
    );
    let reference_nll = mean_nll(&reference.predict_distribution(&dtest).unwrap(), &dtest);
    for direction in [
        DistSplitDirection::Random,
        DistSplitDirection::Cyclic,
        DistSplitDirection::All,
    ] {
        let p = params(Objective::Dist(
            Distributional::new(DistFamily::Normal).with_split_direction(direction),
        ))
        .multi_strategy(MultiStrategy::MultiOutputTree)
        .build()
        .unwrap();
        let model = fit(&p, &dtrain, &dvalid);
        assert_eq!(model.num_trees(), model.num_boost_rounds(), "{direction:?}");
        let dists = model.predict_distribution(&dtest).unwrap();
        let nll = mean_nll(&dists, &dtest);
        assert!(
            (nll - reference_nll).abs() < 0.03,
            "{direction:?}: {nll} vs one tree per parameter {reference_nll}"
        );
        let got = coverage(
            dists.iter().map(|d| d.interval(0.8)),
            dtest.labels().unwrap(),
        );
        assert!((got - 0.8).abs() < 0.03, "{direction:?} coverage {got}");
        let again = train(&p, &dtrain, 20).unwrap();
        assert_eq!(
            again.predict_margin(&dtest).unwrap(),
            train(&p, &dtrain, 20)
                .unwrap()
                .predict_margin(&dtest)
                .unwrap()
        );
        let restored = BoostedModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
        assert_eq!(restored.predict_distribution(&dtest).unwrap(), dists);
    }
    // The random direction follows the seed.
    let with_seed = |seed| {
        let p = params(dist(DistFamily::Normal))
            .multi_strategy(MultiStrategy::MultiOutputTree)
            .seed(seed)
            .build()
            .unwrap();
        train(&p, &dtrain, 10)
            .unwrap()
            .predict_margin(&dtest)
            .unwrap()
    };
    assert_ne!(with_seed(1), with_seed(2));
    // One-parameter families keep scalar trees under the vector strategy.
    let counts = sampled(500, 22, |a, _| Dist::Poisson {
        rate: 1.0 + 5.0 * a,
    });
    let p = params(dist(DistFamily::Poisson))
        .multi_strategy(MultiStrategy::MultiOutputTree)
        .build()
        .unwrap();
    assert_eq!(train(&p, &counts, 5).unwrap().num_trees(), 5);
}
