use super::*;
use crate::objective::gradient_pairs;
use crate::rng::Rng;

/// Margins and labels exercising every family away from the link bounds.
fn cases(family: DistFamily) -> Vec<(Vec<f64>, f64)> {
    match family {
        DistFamily::Normal => vec![
            (vec![0.3, -0.2], 1.1),
            (vec![-2.0, 0.7], -4.5),
            (vec![5.0, 1.5], 5.2),
        ],
        DistFamily::LogNormal => vec![
            (vec![0.3, -0.2], 1.1),
            (vec![-1.0, 0.4], 0.05),
            (vec![2.0, -1.0], 9.0),
        ],
        DistFamily::Gamma => vec![
            (vec![0.3, 0.5], 1.1),
            (vec![-1.0, -0.7], 0.05),
            (vec![2.0, 2.5], 9.0),
            (vec![0.0, 4.0], 1.3),
        ],
        DistFamily::Poisson => vec![(vec![0.3], 2.0), (vec![-1.5], 0.0), (vec![3.0], 25.0)],
        DistFamily::NegativeBinomial => vec![
            (vec![0.3, 0.5], 2.0),
            (vec![-1.0, -1.5], 0.0),
            (vec![2.5, 1.0], 30.0),
            (vec![1.0, 6.0], 1.0),
        ],
    }
}

fn close(a: f64, b: f64, rel: f64, abs: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + abs
}

#[test]
fn gradients_match_finite_differences() {
    // Large enough that the rounding of `ln Γ` differences (the NLL of the
    // count families at large sizes) stays below the tolerance.
    let h = 1e-4;
    for &family in DistFamily::ALL {
        for (eta, y) in cases(family) {
            let g = family.gradient(&eta, y);
            for j in 0..family.n_params() {
                let (mut up, mut down) = (eta.clone(), eta.clone());
                up[j] += h;
                down[j] -= h;
                let fd = (family.nll(&up, y) - family.nll(&down, y)) / (2.0 * h);
                assert!(
                    close(g[j], fd, 1e-6, 1e-7),
                    "{family:?} eta={eta:?} y={y} j={j}: analytic {} vs fd {fd}",
                    g[j]
                );
            }
        }
    }
}

#[test]
fn exact_hessians_match_finite_differences_of_the_gradient() {
    let h = 1e-6;
    for &family in DistFamily::ALL {
        for (eta, y) in cases(family) {
            let hess = family.hessian(&eta, y);
            let k = family.n_params();
            for j in 0..k {
                let (mut up, mut down) = (eta.clone(), eta.clone());
                up[j] += h;
                down[j] -= h;
                let (gu, gd) = (family.gradient(&up, y), family.gradient(&down, y));
                for i in 0..k {
                    let fd = (gu[i] - gd[i]) / (2.0 * h);
                    assert!(
                        close(hess[i][j], fd, 1e-5, 1e-7),
                        "{family:?} eta={eta:?} y={y} H[{i}][{j}]: {} vs fd {fd}",
                        hess[i][j]
                    );
                }
            }
        }
    }
}

/// Monte-Carlo estimates of `E[g gᵀ]` (the Fisher information as the score
/// covariance) and `E[∇² NLL]` agree with the analytic diagonal Fisher, and
/// the off-diagonal score covariance vanishes (the parameterizations are
/// orthogonal). Each check allows five Monte-Carlo standard errors.
#[test]
fn fisher_information_matches_monte_carlo() {
    // Running mean and standard error of a sample.
    #[derive(Default, Clone, Copy)]
    struct Moments {
        sum: f64,
        sum_sq: f64,
    }
    impl Moments {
        fn push(&mut self, x: f64) {
            self.sum += x;
            self.sum_sq += x * x;
        }
        fn mean_se(self, n: usize) -> (f64, f64) {
            let n = n as f64;
            let mean = self.sum / n;
            let var = (self.sum_sq / n - mean * mean).max(0.0);
            (mean, (var / n).sqrt())
        }
    }
    let n = 200_000;
    for &family in DistFamily::ALL {
        for (eta, _) in cases(family) {
            let dist = family.dist_from_margins(&eta);
            let fisher = family.fisher(&eta);
            let k = family.n_params();
            let mut rng = Rng::new(7);
            let mut outer = [[Moments::default(); 2]; 2];
            let mut hess = [Moments::default(); 2];
            for _ in 0..n {
                let y = dist.sample(|| rng.next_u64());
                let g = family.gradient(&eta, y);
                let h = family.hessian(&eta, y);
                for i in 0..k {
                    hess[i].push(h[i][i]);
                    for j in 0..k {
                        outer[i][j].push(g[i] * g[j]);
                    }
                }
            }
            for i in 0..k {
                for (what, m) in [("E[g²]", outer[i][i]), ("E[H]", hess[i])] {
                    let (mean, se) = m.mean_se(n);
                    assert!(
                        (mean - fisher[i]).abs() <= 5.0 * se + 1e-9 * fisher[i],
                        "{family:?} {eta:?} {what}[{i}] = {mean} ± {se} vs Fisher {}",
                        fisher[i]
                    );
                }
            }
            if k == 2 {
                let (mean, se) = outer[0][1].mean_se(n);
                assert!(
                    mean.abs() <= 5.0 * se,
                    "{family:?} {eta:?} E[g0 g1] = {mean} ± {se}"
                );
            }
        }
    }
}

/// The intercept solves the score equations of the marginal fit: the
/// (weighted) gradient sums vanish there, and moving any margin raises the
/// negative log-likelihood.
#[test]
fn intercepts_are_the_marginal_mle() {
    let mut rng = Rng::new(3);
    for &family in DistFamily::ALL {
        let truth = family.dist_from_margins(&cases(family)[0].0);
        let labels: Vec<f32> = (0..4000)
            .map(|_| truth.sample(|| rng.next_u64()) as f32)
            .collect();
        let weights: Vec<f32> = (0..labels.len()).map(|i| 0.5 + (i % 3) as f32).collect();
        for w in [None, Some(weights.as_slice())] {
            let eta = family.mle_margins(&labels, w);
            let weight = |i: usize| w.map_or(1.0, |ws| f64::from(ws[i]));
            let total_nll = |eta: &[f64]| -> f64 {
                labels
                    .iter()
                    .enumerate()
                    .map(|(i, &y)| weight(i) * family.nll(eta, f64::from(y)))
                    .sum()
            };
            let mut score = [0.0f64; 2];
            let mut curvature = [0.0f64; 2];
            for (i, &y) in labels.iter().enumerate() {
                let g = family.gradient(&eta, f64::from(y));
                let f = family.fisher(&eta);
                for j in 0..family.n_params() {
                    score[j] += weight(i) * g[j];
                    curvature[j] += weight(i) * f[j];
                }
            }
            let best = total_nll(&eta);
            for j in 0..family.n_params() {
                // The Newton step from the intercept is negligible.
                assert!(
                    (score[j] / curvature[j]).abs() < 1e-6,
                    "{family:?} weighted={} score[{j}] = {} (curvature {})",
                    w.is_some(),
                    score[j],
                    curvature[j]
                );
                for delta in [-1e-3, 1e-3] {
                    let mut moved = eta.clone();
                    moved[j] += delta;
                    assert!(total_nll(&moved) > best, "{family:?} j={j} delta={delta}");
                }
            }
        }
    }
    // Closed forms: the Normal MLE is the mean and the biased deviation.
    let eta = DistFamily::Normal.mle_margins(&[1.0, 2.0, 6.0], None);
    assert!(close(eta[0], 3.0, 1e-15, 0.0));
    assert!(close(eta[1], (14.0f64 / 3.0).sqrt().ln(), 1e-14, 0.0));
    // Poisson: log of the mean; under-dispersed counts send the negative
    // binomial to its Poisson limit (the size bound).
    let eta = DistFamily::Poisson.mle_margins(&[1.0, 2.0, 6.0], None);
    assert!(close(eta[0], 3f64.ln(), 1e-15, 0.0));
    let eta = DistFamily::NegativeBinomial.mle_margins(&[2.0, 3.0, 2.0, 3.0], None);
    assert!(close(eta[0], 2.5f64.ln(), 1e-15, 0.0));
    assert_eq!(eta[1], LOG_LINK_BOUND);
}

/// `∫ₐᵇ (F(s) - 1{s >= y})² ds` by composite Simpson on panels between
/// `breaks` (plus `y`, `0`, and a geometric grid towards `0⁺` where a Gamma
/// density may be singular), so jumps and kinks fall on panel ends.
fn crps_by_quadrature(dist: &Dist, y: f64, a: f64, b: f64, breaks: &[f64]) -> f64 {
    let f = |s: f64| {
        let step = if s >= y { 1.0 } else { 0.0 };
        (dist.cdf(s) - step).powi(2)
    };
    let mut points: Vec<f64> = vec![a, b, y, 0.0];
    points.extend((1..=14).map(|e| 10f64.powi(-e)));
    points.extend_from_slice(breaks);
    points.retain(|&p| (a..=b).contains(&p));
    points.sort_by(f64::total_cmp);
    points.dedup();
    points
        .windows(2)
        .map(|w| {
            // Evaluate strictly inside the panel: the integrand is
            // right-continuous but its panel limits are what matter.
            let (lo, hi) = (w[0], w[1]);
            let n = 2000;
            let h = (hi - lo) / f64::from(n);
            let inner = |s: f64| f(s.clamp(lo + 1e-13 * (hi - lo), hi - 1e-13 * (hi - lo)));
            let mut sum = inner(lo) + inner(hi);
            for i in 1..n {
                let w = if i % 2 == 1 { 4.0 } else { 2.0 };
                sum += w * inner(lo + f64::from(i) * h);
            }
            sum * h / 3.0
        })
        .sum()
}

#[test]
fn closed_form_crps_matches_numeric_integration() {
    let continuous = [
        (
            Dist::Normal {
                mu: 1.0,
                sigma: 2.0,
            },
            [-3.0, 1.0, 4.5],
        ),
        (
            Dist::LogNormal {
                mu: 0.2,
                sigma: 0.6,
            },
            [0.3, 1.2, 4.0],
        ),
        (
            Dist::Gamma {
                mean: 2.0,
                shape: 3.0,
            },
            [0.2, 1.8, 7.0],
        ),
        (
            Dist::Gamma {
                mean: 1.5,
                shape: 0.7,
            },
            [0.05, 1.0, 5.0],
        ),
    ];
    for (dist, ys) in continuous {
        let (lo, hi) = (dist.quantile(1e-12).min(-0.0), dist.quantile(1.0 - 1e-12));
        for y in ys {
            let numeric = crps_by_quadrature(&dist, y, lo.min(y) - 1.0, hi.max(y) + 1.0, &[]);
            assert!(
                close(dist.crps(y), numeric, 1e-6, 1e-8),
                "{dist:?} y={y}: closed {} vs numeric {numeric}",
                dist.crps(y)
            );
        }
        // Outside the positive support the CRPS is E|X - y| - E|X - X'|/2.
        if !matches!(dist, Dist::Normal { .. }) {
            let numeric = crps_by_quadrature(&dist, -0.5, -0.5, hi + 1.0, &[]);
            assert!(close(dist.crps(-0.5), numeric, 1e-6, 1e-8), "{dist:?}");
        }
    }
}

#[test]
fn count_crps_matches_the_integral_of_the_step_cdf() {
    for dist in [
        Dist::Poisson { rate: 3.5 },
        Dist::NegativeBinomial {
            mean: 4.0,
            size: 1.5,
        },
        Dist::Poisson { rate: 400.0 },
    ] {
        let hi = dist.quantile(1.0 - 1e-13) + 2.0;
        for y in [-1.5, 0.0, 2.0, 3.4, dist.mean().round()] {
            // Panels end at the integers, where the step CDF jumps.
            let steps: Vec<f64> = (0..=hi as u32).map(f64::from).collect();
            let numeric = crps_by_quadrature(&dist, y, y.min(0.0) - 1.0, hi, &steps);
            assert!(
                close(dist.crps(y), numeric, 1e-6, 1e-8),
                "{dist:?} y={y}: {} vs {numeric}",
                dist.crps(y)
            );
        }
    }
}

/// `E|X - y| - E|X - X'|/2` of a count distribution from its pmf over
/// `lo..=hi`, normalized there (independent of the CRPS step sum):
/// `E|X - y|` summed directly and `E|X - X'|/2 = Σ_k F(k)(1 - F(k))`.
fn count_crps_by_expectation(dist: &Dist, y: f64, lo: u64, hi: u64) -> f64 {
    let pmf: Vec<f64> = (lo..=hi).map(|k| dist.log_prob(k as f64).exp()).collect();
    let total: f64 = pmf.iter().sum();
    let (mut abs_dev, mut spread, mut cdf) = (0.0, 0.0, 0.0);
    for (i, p) in pmf.iter().enumerate() {
        let p = p / total;
        abs_dev += p * ((lo + i as u64) as f64 - y).abs();
        cdf += p;
        spread += cdf * (1.0 - cdf);
    }
    abs_dev - spread
}

/// Labels far outside the numerical support: the unit steps between the
/// support and `y` count in full (none are dropped after a step budget), and
/// supports wider than the step budget are summed in blocks instead of
/// being cut short.
#[test]
fn count_crps_handles_residuals_far_outside_the_support() {
    let cases = [
        (Dist::Poisson { rate: 1.0 }, 1e6, 60),
        (Dist::Poisson { rate: 1.0 }, 1e6 + 0.25, 60),
        (
            Dist::NegativeBinomial {
                mean: 3.0,
                size: 2.0,
            },
            1e7,
            400,
        ),
        (Dist::Poisson { rate: 1e6 }, 0.0, 1_020_000),
        (Dist::Poisson { rate: 1e6 }, 3e6, 1_020_000),
        // 24 standard deviations span more than the step budget.
        (Dist::Poisson { rate: 1e9 }, 0.0, 1_000_400_000),
        (Dist::Poisson { rate: 1e9 }, 1e9 + 2e4, 1_000_400_000),
        (
            Dist::NegativeBinomial {
                mean: 2e5,
                size: 40.0,
            },
            5e3,
            800_000,
        ),
    ];
    for (dist, y, hi) in cases {
        // Below 14 standard deviations under the mean the mass is < 1e-40.
        let lo = (dist.mean() - 14.0 * dist.std_dev()).max(0.0) as u64;
        let reference = count_crps_by_expectation(&dist, y, lo, hi);
        let got = dist.crps(y);
        assert!(
            close(got, reference, 1e-7, 1e-6),
            "{dist:?} y={y}: {got} vs {reference}"
        );
        // The CRPS is at least the distance to the mean minus E|X - X'|/2.
        assert!(got >= (y - dist.mean()).abs() - dist.std_dev(), "{dist:?}");
    }
}

#[test]
fn quantiles_invert_the_cdf() {
    let continuous = [
        Dist::Normal {
            mu: -1.0,
            sigma: 0.5,
        },
        Dist::LogNormal {
            mu: 1.0,
            sigma: 1.3,
        },
        Dist::Gamma {
            mean: 3.0,
            shape: 0.3,
        },
        Dist::Gamma {
            mean: 3.0,
            shape: 50.0,
        },
    ];
    for dist in continuous {
        for p in [1e-8, 0.01, 0.2, 0.5, 0.9, 0.999_999] {
            let q = dist.quantile(p);
            assert!(close(dist.cdf(q), p, 1e-9, 1e-15), "{dist:?} p={p} q={q}");
        }
        assert_eq!(dist.quantile(1.0), f64::INFINITY);
    }
    let counts = [
        Dist::Poisson { rate: 0.2 },
        Dist::Poisson { rate: 1e4 },
        Dist::NegativeBinomial {
            mean: 7.0,
            size: 0.4,
        },
    ];
    for dist in counts {
        for p in [0.0, 1e-6, 0.3, 0.5, 0.95, 0.999_999] {
            let q = dist.quantile(p);
            assert_eq!(q, q.round(), "{dist:?}");
            assert!(dist.cdf(q) >= p, "{dist:?} p={p} q={q}");
            assert!(q == 0.0 || dist.cdf(q - 1.0) < p, "{dist:?} p={p} q={q}");
        }
    }
}

#[test]
fn counts_are_normalized_and_consistent_with_the_cdf() {
    for dist in [
        Dist::Poisson { rate: 6.0 },
        Dist::NegativeBinomial {
            mean: 6.0,
            size: 2.0,
        },
    ] {
        let mut cdf = 0.0;
        for k in 0..200 {
            let k = f64::from(k);
            cdf += dist.log_prob(k).exp();
            assert!(close(dist.cdf(k), cdf, 1e-10, 1e-14), "{dist:?} k={k}");
        }
        assert!(close(cdf, 1.0, 1e-12, 0.0));
        assert_eq!(dist.log_prob(-1.0), f64::NEG_INFINITY);
    }
}

/// One distribution per family, away from the parameter bounds.
fn one_per_family() -> [Dist; 5] {
    [
        Dist::Normal {
            mu: 2.0,
            sigma: 3.0,
        },
        Dist::LogNormal {
            mu: 0.1,
            sigma: 0.4,
        },
        Dist::Gamma {
            mean: 5.0,
            shape: 2.0,
        },
        Dist::Poisson { rate: 4.0 },
        Dist::NegativeBinomial {
            mean: 4.0,
            size: 3.0,
        },
    ]
}

#[test]
fn sampling_is_seeded_and_matches_the_moments() {
    for dist in one_per_family() {
        let draw = |seed| {
            let mut rng = Rng::new(seed);
            (0..50_000)
                .map(|_| dist.sample(|| rng.next_u64()))
                .collect::<Vec<f64>>()
        };
        let a = draw(11);
        assert_eq!(a, draw(11), "{dist:?}: same seed, same stream");
        assert_ne!(a, draw(12), "{dist:?}");
        let n = a.len() as f64;
        let mean = a.iter().sum::<f64>() / n;
        let var = a.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let se = (dist.variance() / n).sqrt();
        assert!(
            (mean - dist.mean()).abs() < 5.0 * se,
            "{dist:?} mean {mean}"
        );
        assert!(close(var, dist.variance(), 0.05, 0.0), "{dist:?} var {var}");
    }
}

#[test]
fn poisson_matches_count_poisson_without_max_delta_step() {
    let preds = [0.3f32, -1.0, 2.5, 0.0];
    let labels = [2.0f32, 0.0, 14.0, 1.0];
    let weights = [1.0f32, 0.5, 2.0, 1.5];
    let dist = DistObjective::new(DistFamily::Poisson, DistGradient::Fisher);
    let count = crate::objective::Poisson::new(0.0);
    for w in [None, Some(weights.as_slice())] {
        let a = gradient_pairs(&dist, &preds, &labels, w);
        let b = gradient_pairs(&count, &preds, &labels, w);
        for (x, y) in a.iter().zip(&b) {
            assert!(close(f64::from(x.grad), f64::from(y.grad), 1e-6, 1e-7));
            assert!(close(f64::from(x.hess), f64::from(y.hess), 1e-6, 1e-7));
        }
    }
}

#[test]
fn gradient_modes_pair_the_gradient_with_the_selected_curvature() {
    let preds = [0.4f32, 0.3, -0.2, 1.1];
    let labels = [1.5f32, 0.2];
    let weights = [2.0f32, 0.5];
    let family = DistFamily::Gamma;
    for mode in [
        DistGradient::Fisher,
        DistGradient::Hessian,
        DistGradient::Natural,
    ] {
        let objective = DistObjective::new(family, mode);
        let out = gradient_pairs(&objective, &preds, &labels, Some(&weights));
        for i in 0..2 {
            let eta = [f64::from(preds[2 * i]), f64::from(preds[2 * i + 1])];
            let y = f64::from(labels[i]);
            let w = f64::from(weights[i]);
            let g = family.gradient(&eta, y);
            let fisher = family.fisher(&eta);
            let hess = family.hessian(&eta, y);
            for j in 0..2 {
                let (eg, eh) = match mode {
                    DistGradient::Fisher => (g[j], fisher[j]),
                    DistGradient::Hessian => (g[j], hess[j][j].max(MIN_CURVATURE)),
                    DistGradient::Natural => (g[j] / fisher[j], 1.0),
                };
                let pair = out[2 * i + j];
                assert!(close(f64::from(pair.grad), w * eg, 1e-6, 1e-9), "{mode:?}");
                assert!(close(f64::from(pair.hess), w * eh, 1e-6, 1e-9), "{mode:?}");
            }
        }
    }
    // The exact Hessian of the negative-binomial size turns negative for
    // counts far above the mean; the floor keeps it a valid (positive) tree
    // statistic.
    let family = DistFamily::NegativeBinomial;
    let eta = [0.0, 3.0];
    assert!(family.hessian(&eta, 5.0)[1][1] < 0.0);
    let objective = DistObjective::new(family, DistGradient::Hessian);
    let out = gradient_pairs(&objective, &[0.0, 3.0], &[5.0], None);
    assert_eq!(out[1].hess, crate::objective::MIN_HESS);
}

#[test]
fn transforms_and_links_round_trip() {
    let objective = DistObjective::new(DistFamily::Normal, DistGradient::Fisher);
    let mut preds = [1.5f32, 0.25f32.ln(), -2.0, 50.0];
    objective.pred_transform(&mut preds);
    assert_eq!(preds[0], 1.5);
    assert!((preds[1] - 0.25).abs() < 1e-7);
    // Log links are clamped: exp(30), not exp(50).
    assert_eq!(preds[3], (LOG_LINK_BOUND.exp()) as f32);
    let mut scores = [1.5f32, 0.25];
    objective.probs_to_margins(&mut scores);
    assert!((scores[1] - 0.25f32.ln()).abs() < 1e-7);
    assert_eq!(scores[0], 1.5);
    let mut bad = [0.0f32, -1.0];
    objective.probs_to_margins(&mut bad);
    assert!(bad[1].is_nan());
    for &family in DistFamily::ALL {
        assert_eq!(
            DistFamily::from_objective(family.objective_name()),
            Some(family)
        );
    }
    assert!(DistFamily::from_objective("reg:squarederror").is_none());
}

#[test]
fn dist_new_validates_parameters() {
    assert!(Dist::new(DistFamily::Normal, &[0.0, 1.0]).is_ok());
    assert!(Dist::new(DistFamily::Normal, &[0.0, 0.0]).is_err());
    assert!(Dist::new(DistFamily::Normal, &[0.0]).is_err());
    assert!(Dist::new(DistFamily::Poisson, &[f64::NAN]).is_err());
    assert!(Dist::new(DistFamily::Gamma, &[1.0, -2.0]).is_err());
    let d = Dist::new(DistFamily::NegativeBinomial, &[3.0, 2.0]).unwrap();
    assert_eq!(d.params(), vec![3.0, 2.0]);
    assert_eq!(d.family(), DistFamily::NegativeBinomial);
    assert!(close(d.variance(), 3.0 + 4.5, 1e-15, 0.0));
    let (lo, hi) = Dist::Normal {
        mu: 0.0,
        sigma: 1.0,
    }
    .interval(0.95);
    assert!(close(hi, 1.959_963_984_540_054, 1e-12, 0.0) && close(lo, -hi, 1e-12, 0.0));
}

#[test]
fn shared_trees_split_on_one_parameter_column() {
    let gpair: Vec<GradPair> = (0..6)
        .map(|i| GradPair::new(i as f32, 10.0 + i as f32))
        .collect();
    let plain = DistObjective::new(DistFamily::Gamma, DistGradient::Fisher);
    assert_eq!(plain.split_gradient(0, &gpair), None);
    let all = plain.with_split_direction(DistSplitDirection::All, 0);
    assert_eq!(all.split_gradient(0, &gpair), None);
    let cyclic = plain.with_split_direction(DistSplitDirection::Cyclic, 0);
    for iteration in 0..4 {
        let m = iteration % 2;
        let split = cyclic.split_gradient(iteration, &gpair).unwrap();
        assert_eq!(split.n_targets, 1);
        let column: Vec<GradPair> = gpair.iter().skip(m).step_by(2).copied().collect();
        assert_eq!(split.gpair, column);
    }
    // Random: a function of (seed, iteration) that visits every parameter
    // about equally often.
    let draws = |seed| -> Vec<usize> {
        let random = plain.with_split_direction(DistSplitDirection::Random, seed);
        (0..2000)
            .map(|t| random.split_parameter(t).unwrap())
            .collect()
    };
    let first = draws(5);
    assert_eq!(first, draws(5));
    assert_ne!(first, draws(6));
    let ones = first.iter().filter(|&&m| m == 1).count();
    assert!((900..1100).contains(&ones), "{ones} of 2000");
    // One parameter: nothing to reduce.
    let poisson = DistObjective::new(DistFamily::Poisson, DistGradient::Fisher)
        .with_split_direction(DistSplitDirection::Random, 0);
    assert_eq!(poisson.split_gradient(0, &gpair[..3]), None);
}

// Parameters where the formulas are prone to cancellation, overflow, or
// early termination. References are high-precision mpmath (40 digits) or
// scipy evaluations of the stated quantities.

/// A Gamma label far below its mean: `t - 1` rounds to `-1`, so the log gap
/// `t - 1 - ln t` (≈ 45.05 for `t = 1e-20`) must not be formed from it and
/// come out `+∞`.
#[test]
fn gamma_gradients_stay_finite_for_tiny_label_ratios() {
    let eta = [0.0, 0.0];
    let y = 1e-20;
    let g = DistFamily::Gamma.gradient(&eta, y);
    // a (ψ(a) - ln a + t - 1 - ln t) at m = a = 1.
    assert!(close(g[1], 44.474_486_194_979_38, 1e-14, 0.0), "{g:?}");
    let h = DistFamily::Gamma.hessian(&eta, y);
    assert!(h.iter().flatten().all(|v| v.is_finite()), "{h:?}");
    assert!(DistFamily::Gamma.nll(&eta, y).is_finite());
}

/// A zero count under a large mean at the smallest size: `(y - m)/(r + m)`
/// rounds to `-1`, yet the size score must stay finite so the intercept
/// reaches the MLE (NLL 12.88) rather than the `ln r = -30` bound (NLL
/// 39.21).
#[test]
fn negative_binomial_intercept_survives_a_zero_count_under_a_large_mean() {
    let family = DistFamily::NegativeBinomial;
    assert!(nb_size_score(5000.0, (-LOG_LINK_BOUND).exp(), 0.0).is_finite());
    let eta = family.mle_margins(&[0.0, 10_000.0], None);
    assert!(close(eta[0], 5000f64.ln(), 1e-15, 0.0), "{eta:?}");
    assert!(close(eta[1], -2.518_424_656_738_833, 0.0, 1e-8), "{eta:?}");
    let nll = family.nll(&eta, 0.0) + family.nll(&eta, 10_000.0);
    assert!(close(nll, 12.884_983_486_131_556, 1e-10, 0.0), "{nll}");
    for y in [0.0, 10_000.0] {
        let g = family.gradient(&eta, y);
        assert!(g.iter().all(|v| v.is_finite()), "{g:?}");
    }
}

/// Wide negative binomials: the size Fisher sum covers the whole support of
/// ~340 000 values (`mean = size = 1e8`), mode and tail included, and stays
/// positive (not a negative entry floored to `1e-16`). References: the
/// survival series `Σ_k P(Y > k)/(r + k)²` summed exactly over the whole
/// support.
#[test]
fn negative_binomial_size_fisher_covers_wide_supports() {
    for (mean, size, reference) in [
        (1e8, 1e8, 0.124_999_999_583_333_34),
        // Sharp head at zero (size below one), 24 sd ≈ 340 000 values.
        (1e4, 0.5, 0.722_643_525_822_104),
    ] {
        let i = DistFamily::NegativeBinomial.fisher(&[f64::ln(mean), f64::ln(size)]);
        assert!(close(i[1], reference, 1e-6, 0.0), "{mean} {size}: {i:?}");
    }
}

/// A skewed wide negative binomial: the blocks must keep the mass near zero
/// that sd-width blocks evaluated at their midpoint would lose (CRPS
/// 2876.10 at `y = 0`). References: scipy's CDF summed over single steps up
/// to 6e6.
#[test]
fn count_crps_keeps_the_head_of_skewed_wide_counts() {
    let dist = Dist::NegativeBinomial {
        mean: 1e4,
        size: 0.1,
    };
    for (y, reference) in [
        (0.0, 1_168.446_191_399_923_5),
        (2.5, 1_167.639_588_097_035_4),
        (1e4, 6_278.362_015_659_435),
        (1e5, 84_559.396_877_837_03),
    ] {
        let got = dist.crps(y);
        assert!(close(got, reference, 1e-6, 0.0), "y={y}: {got}");
    }
}

/// Gamma at the shape bound `e^30` (what constant labels fit): the CDF,
/// quantiles, CRPS, and log density keep full precision.
#[test]
fn gamma_handles_the_largest_shape() {
    let a = LOG_LINK_BOUND.exp();
    let dist = Dist::Gamma {
        mean: 1.0,
        shape: a,
    };
    assert!(close(dist.cdf(1.0), 0.500_000_040_679_123_1, 1e-9, 0.0));
    // 2 a^a e^-a / Γ(a + 1) - Γ(a + 1/2) / (√π Γ(a) a).
    let crps = dist.crps(1.0);
    assert!(close(crps, 7.148_783_583_195_937e-8, 1e-6, 0.0), "{crps}");
    // a ln a - ln Γ(a) - a.
    let lp = dist.log_prob(1.0);
    assert!(close(lp, 14.081_061_466_795_32, 1e-13, 0.0), "{lp}");
    for p in [1e-6, 0.3, 0.5, 0.999] {
        let q = dist.quantile(p);
        assert!(close(dist.cdf(q), p, 1e-6, 1e-9), "p={p} q={q}");
    }
}

/// A large log standard deviation: `2Φ(σ/√2) - 1` rounds to one, yet the
/// LogNormal CRPS must come out `≈ 4.0e14`, not 0.
#[test]
fn lognormal_crps_keeps_the_upper_tail() {
    let dist = Dist::LogNormal {
        mu: 0.0,
        sigma: 12.0,
    };
    // y (2Φ(w) - 1) + 2 e^72 (Φ(-12/√2) - Φ(w - 12)), w = ln y / 12.
    assert!(close(dist.crps(1.0), 399_981_265_934_752.7, 1e-12, 0.0));
    // 2 e^72 Φ(-12/√2) - y.
    assert!(close(dist.crps(-1.0), 399_981_265_934_753.8, 1e-12, 0.0));
    // The scale overflows f64 alone, the CRPS does not.
    let wide = Dist::LogNormal {
        mu: 0.0,
        sigma: 40.0,
    };
    assert!(wide.crps(1.0).is_finite() && wide.crps(1.0) > 0.0);
}

/// A tail probability below the smallest double under a huge scale:
/// `e^{756.25} Φ(-55/√2)` with `Φ(-38.9) ≈ 1e-331` must be evaluated in log
/// space, not as the logarithm of a CDF that underflowed to zero (CRPS 0 at
/// `y = 0`). References: the closed form in mpmath (60 digits).
#[test]
fn lognormal_crps_survives_an_underflowing_tail_probability() {
    let dist = Dist::LogNormal {
        mu: -756.25,
        sigma: 55.0,
    };
    for (y, reference) in [
        (0.0, 0.020_502_447_384_614_8),
        (1e-300, 0.020_502_447_384_614_8),
        (1.0, 1.020_502_447_384_614_7),
    ] {
        let got = dist.crps(y);
        assert!(close(got, reference, 1e-12, 0.0), "y={y}: {got}");
    }
}

/// Negative binomials near their Poisson limit (`size = 1e13`, and the f32
/// rounding of the `e^30` bound that predictions carry): the log-gamma
/// differences in the log pmf and the incomplete-beta prefactor must not
/// cancel (a cancelling evaluation is off by 0.06 in the log pmf and gives
/// `cdf(1)` 0.325 instead of 0.287).
#[test]
fn negative_binomial_near_the_poisson_limit() {
    for (size, ln_p1, cdf1, cdf2) in [
        (
            1e13,
            -1.583_709_268_125_782_5,
            0.287_297_495_183_684_25,
            0.543_813_115_883_345_6,
        ),
        (
            f64::from(LOG_LINK_BOUND.exp() as f32),
            -1.583_709_268_125_786_5,
            0.287_297_495_183_681_8,
            0.543_813_115_883_344_5,
        ),
    ] {
        let dist = Dist::NegativeBinomial { mean: 2.5, size };
        assert!(close(dist.log_prob(1.0), ln_p1, 0.0, 1e-12), "{size}");
        assert!(close(dist.cdf(1.0), cdf1, 1e-12, 0.0), "{size}");
        assert!(close(dist.cdf(2.0), cdf2, 1e-12, 0.0), "{size}");
        assert_eq!(dist.quantile(0.3), 2.0, "{size}");
    }
}

/// The extreme random words map strictly inside `(0, 1)`: `u64::MAX` must
/// not round to `u = 1` and give an infinite sample.
#[test]
fn sampling_stays_finite_at_the_extreme_rng_words() {
    for dist in one_per_family() {
        for word in [0, u64::MAX] {
            let s = dist.sample(|| word);
            assert!(s.is_finite(), "{dist:?} word={word:#x}: {s}");
        }
    }
}

/// The count CDFs at `+∞` (and so at `quantile(1)`) are one, not `NaN`.
#[test]
fn count_cdfs_are_one_at_infinity() {
    for dist in [
        Dist::Poisson { rate: 3.5 },
        Dist::NegativeBinomial {
            mean: 4.0,
            size: 1.5,
        },
    ] {
        assert_eq!(dist.cdf(f64::INFINITY), 1.0, "{dist:?}");
        assert_eq!(dist.cdf(dist.quantile(1.0)), 1.0, "{dist:?}");
    }
}

/// Count quantiles beyond `2^53`, where adjacent floats are more than 1
/// apart, still terminate (the bisection once looped forever when the
/// floored midpoint fell back onto `lo`) and bracket `p`.
#[test]
fn count_quantiles_terminate_past_integer_precision() {
    let poisson = Dist::new(DistFamily::Poisson, &[1e20]).unwrap();
    let median = poisson.quantile(0.5);
    assert!((median - 1e20).abs() < 1e11, "{median}");
    for dist in [
        poisson,
        Dist::new(DistFamily::NegativeBinomial, &[1e20, 1e3]).unwrap(),
    ] {
        for p in [1e-6, 0.3, 0.5, 0.9] {
            let q = dist.quantile(p);
            assert!(q.is_finite() && dist.cdf(q) >= p, "{dist:?} p={p}: {q}");
        }
    }
}

/// A Gamma quantile far in the lower tail: the Halley iteration must not
/// start from a floor such as `1e-3` (100 steps from there stop at
/// `1.9e-51`, CDF `1.9e-102`). For shape 2, `P(2, x) = x²/2 (1 + O(x))`, so
/// the quantile is `√2 · 1e-150`.
#[test]
fn gamma_quantiles_reach_the_deep_lower_tail() {
    let dist = Dist::Gamma {
        mean: 2.0,
        shape: 2.0,
    };
    let q = dist.quantile(1e-300);
    assert!(close(q, 1.414_213_562_373_095e-150, 1e-12, 0.0), "{q}");
    assert!(close(dist.cdf(q), 1e-300, 1e-11, 0.0), "{}", dist.cdf(q));
}
